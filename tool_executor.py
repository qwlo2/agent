"""工具注册及执行框架，默认提供面向旅游规划的高德地图工具。"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import urlopen

from dotenv import dotenv_values


class ToolExecutor:
    """统一管理工具描述与 Python 实现，按工具名称执行调用。

    默认注册高德地点搜索、周边搜索和路线查询，也可注册其他来源的工具。
    工具执行结果直接返回，由 Agent 负责序列化并关联 tool_call_id。
    """

    def __init__(self, env_file: str | Path | None = None) -> None:
        """加载高德配置并注册内置工具，不在初始化时发送请求。

        Args:
            env_file: 配置文件路径，默认使用当前模块同目录下的 .env。
                同名系统环境变量优先于文件配置。

        高德 Key 在实际请求时校验，未填写时仍可注册和执行其他工具。
        """
        config = {
            **dotenv_values(env_file or Path(__file__).with_name(".env")),
            **os.environ,
        }
        self._amap_key = (config.get("AMAP_API_KEY") or "").strip()
        self._timeout = float(config.get("AMAP_TIMEOUT") or "15")
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

        pagination = {
            "page": {"type": "integer", "minimum": 1, "description": "页码，默认 1"},
            "page_size": {
                "type": "integer", "minimum": 1, "maximum": 25,
                "description": "每页条数，默认 10，最多 25",
            },
        }
        self.register_tool(
            name="amap_search_places",
            description="按城市和关键词检索景点、餐厅、酒店等地点，获取地址及高德坐标。",
            parameters={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "地点名称或类别，如西湖、博物馆、酒店"},
                    "city": {"type": "string", "description": "搜索城市名称或 adcode，如杭州"},
                    **pagination,
                },
                "required": ["keywords", "city"],
                "additionalProperties": False,
            },
            handler=self.search_places,
        )
        self.register_tool(
            name="amap_search_nearby",
            description="检索指定地点周边的餐厅、酒店、景点等，按距离排序。坐标应来自高德搜索结果，不要编造。",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "高德 GCJ-02 坐标，格式为经度,纬度"},
                    "keywords": {"type": "string", "description": "搜索关键词，如餐厅、酒店、景点"},
                    "radius": {
                        "type": "integer", "minimum": 1, "maximum": 50000,
                        "description": "搜索半径，单位米，默认 1000",
                    },
                    **pagination,
                },
                "required": ["location", "keywords"],
                "additionalProperties": False,
            },
            handler=self.search_nearby,
        )
        self.register_tool(
            name="amap_plan_route",
            description="查询两个地点间的步行、驾车或公交路线及预计距离和耗时。起终点应使用高德搜索返回的坐标。",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "起点高德 GCJ-02 坐标，经度,纬度"},
                    "destination": {"type": "string", "description": "终点高德 GCJ-02 坐标，经度,纬度"},
                    "mode": {
                        "type": "string", "enum": ["walking", "driving", "transit"],
                        "description": "出行方式，默认 walking；transit 包含公交与地铁",
                    },
                    "city": {"type": "string", "description": "公交模式必填，起点城市名称或 citycode"},
                    "destination_city": {"type": "string", "description": "跨城公交必填，终点城市名称或 citycode"},
                },
                "required": ["origin", "destination"],
                "additionalProperties": False,
            },
            handler=self.plan_route,
        )

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """注册工具描述和实现，供模型选择及执行器调用。

        Args:
            name: 唯一工具名称，模型调用时通过此名称查找实现。
            description: 面向模型的工具用途说明。
            parameters: 参数的 JSON Schema；字段名应与 handler 的参数一致。
            handler: 接收关键字参数的同步函数，结果应可序列化为 JSON。

        Raises:
            ValueError: 工具名称已注册。
        """
        if name in self._tools:
            raise ValueError(f"工具已注册：{name}")
        definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": deepcopy(parameters),
            },
        }
        self._tools[name] = (definition, handler)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """返回所有工具描述的副本，可直接传入 Trip_Client.ask(tools=...)。

        返回值不包含 Python 函数，修改返回内容不会影响已注册工具。
        """
        return deepcopy([definition for definition, _ in self._tools.values()])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """按名称执行一次工具调用并原样返回结果，不自动捕获执行异常。

        Args:
            name: 模型返回的 ToolCall.name。
            arguments: 模型返回的 ToolCall.arguments，作为关键字参数传给工具。

        Raises:
            KeyError: 工具名称未注册。
            TypeError: 参数与工具函数签名不匹配。
        """
        _, handler = self._tools[name]
        return handler(**arguments)

    def search_places(
        self, keywords: str, city: str, page: int = 1, page_size: int = 10,
    ) -> dict[str, Any]:
        """在指定城市检索旅游地点，返回高德原始 JSON（含 pois 列表）。

        Args:
            keywords: 地点名称或类别，例如西湖、餐厅、酒店。
            city: 城市名称或 adcode，搜索限制在该城市内。
            page: 页码，默认 1。
            page_size: 每页条数，默认 10，建议不超过 25。
        """
        return self._request("/v3/place/text", {
            "keywords": keywords, "city": city, "citylimit": "true",
            "page": page, "offset": page_size, "extensions": "all",
        })

    def search_nearby(
        self, location: str, keywords: str, radius: int = 1000,
        page: int = 1, page_size: int = 10,
    ) -> dict[str, Any]:
        """检索中心点周边的旅游配套设施，返回按距离排序的高德原始 JSON。

        Args:
            location: 中心点 GCJ-02 坐标，经度在前、纬度在后，以逗号分隔。
            keywords: 地点名称或类别，例如餐厅、酒店、景点。
            radius: 搜索半径，单位米，默认 1000，最大 50000。
            page: 页码，默认 1。
            page_size: 每页条数，默认 10，建议不超过 25。
        """
        return self._request("/v3/place/around", {
            "location": location, "keywords": keywords, "radius": radius,
            "page": page, "offset": page_size, "sortrule": "distance",
            "extensions": "all",
        })

    def plan_route(
        self, origin: str, destination: str, mode: str = "walking",
        city: str | None = None, destination_city: str | None = None,
    ) -> dict[str, Any]:
        """查询行程两站之间的交通方案，返回含 route 的高德原始 JSON。

        Args:
            origin: 起点 GCJ-02 坐标，格式为经度,纬度。
            destination: 终点 GCJ-02 坐标，格式为经度,纬度。
            mode: walking、driving 或 transit，默认步行。
            city: 公交模式必填，起点城市名称或 citycode。
            destination_city: 跨城公交的终点城市，市内公交可省略。

        Raises:
            ValueError: 出行方式不支持，或公交模式未提供起点城市。
        """
        routes = {
            "walking": "/v3/direction/walking",
            "driving": "/v3/direction/driving",
            "transit": "/v3/direction/transit/integrated",
        }
        if mode not in routes:
            raise ValueError(f"不支持的出行方式：{mode}")
        params: dict[str, Any] = {"origin": origin, "destination": destination}
        if mode == "transit":
            if not city:
                raise ValueError("公交路线查询需要提供 city")
            params["city"] = city
            if destination_city is not None:
                params["cityd"] = destination_city
        return self._request(routes[mode], params)

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送一次高德 GET 请求，在此统一检查 Key 和业务响应状态。

        Args:
            path: 高德 API 路径，由内置工具指定。
            params: 不含 Key 的业务参数，统一编码为查询字符串。

        Returns:
            高德返回的完整 JSON 字典，网络或 JSON 解析异常向调用方传播。

        Raises:
            ValueError: 未设置 AMAP_API_KEY。
            RuntimeError: 高德返回业务错误，包含 info 和 infocode。
        """
        if not self._amap_key:
            raise ValueError("请在 .env 或系统环境变量中设置 AMAP_API_KEY（Web 服务类型）")
        query = urlencode({**params, "key": self._amap_key, "output": "JSON"})
        with urlopen(f"https://restapi.amap.com{path}?{query}", timeout=self._timeout) as response:
            data = json.load(response)
        if data.get("status") != "1":
            raise RuntimeError(f"高德 API 错误：{data.get('info')}（{data.get('infocode')}）")
        return data
