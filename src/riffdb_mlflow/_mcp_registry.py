from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

from mlflow.entities.mcp_access_endpoint import MCPAccessEndpoint
from mlflow.entities.mcp_server import (
    VALID_STATUS_TRANSITIONS,
    MCPRemoteTransportType,
    MCPServer,
    MCPStatus,
    MCPTool,
    validate_mcp_server_name,
)
from mlflow.entities.mcp_server_version import ConnectOptionSettings, MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import (
    NOT_SET,
    MCPIcon,
)
from mlflow.utils.search_utils import (
    SearchMCPAccessEndpointUtils,
    SearchMCPServerUtils,
    SearchMCPServerVersionUtils,
    SearchUtils,
)
from mlflow.utils.semver_utils import encode_prerelease_sort_key, parse_semver
from mlflow.utils.validation import (
    _strip_mcp_icon_response_fields,
    _validate_mcp_icon_payloads,
    _validate_mcp_initial_status,
    _validate_mcp_tool_payloads,
)

from riffdb_mlflow._gateway_and_scorers import RiffDBGatewayAndScorerMixin
from riffdb_mlflow.generated.client import (
    CreateMCPServerEntityInput,
    CreateMCPServerEntityMCPServerCreated,
    CreateMCPServerEntityMCPServerExists,
    CreateMCPServerVersionEntityInput,
    CreateMCPServerVersionEntityMCPServerVersionCreated,
    CreateMCPServerVersionEntityMCPServerVersionExists,
    DeleteMCPAccessEndpointEntityInput,
    DeleteMCPAccessEndpointEntityMCPAccessEndpointDeleted,
    DeleteMCPAccessEndpointEntityMCPAccessEndpointMissing,
    DeleteMCPServerEntityInput,
    DeleteMCPServerEntityMCPServerDeleted,
    DeleteMCPServerEntityMCPServerMissing,
    GetMCPAccessEndpointFound,
    GetMCPAccessEndpointMissing,
    GetMCPAccessEndpointParams,
    GetMCPServerFound,
    GetMCPServerMissing,
    GetMCPServerParams,
    GetMCPServerVersionFound,
    GetMCPServerVersionMissing,
    GetMCPServerVersionParams,
    ListMCPAccessEndpointsByServerFound,
    ListMCPAccessEndpointsByServerParams,
    ListMCPServerVersionsByServerFound,
    ListMCPServerVersionsByServerParams,
    MCPAccessEndpointEntity,
    MCPServerEntity,
    MCPServerVersionEntity,
    SearchMCPAccessEndpointsFound,
    SearchMCPAccessEndpointsParams,
    SearchMCPServersFound,
    SearchMCPServersParams,
    UpsertMCPAccessEndpointEntityInput,
    UpsertMCPAccessEndpointEntityMCPAccessEndpointUpserted,
    UpsertMCPServerEntityInput,
    UpsertMCPServerEntityMCPServerUpserted,
    UpsertMCPServerVersionEntityInput,
    UpsertMCPServerVersionEntityMCPServerVersionUpserted,
)

_SCOPE = "global"
_PAGE_BATCH_SIZE = 10
_NOT_SET: Any = NOT_SET
SEARCH_MCP_SERVER_MAX_RESULTS_THRESHOLD = 1000
_VALID_FILTER_COMPARATORS = {"=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _pack_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _unpack_dict(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload.decode("utf-8"))
    if isinstance(loaded, dict):
        return loaded
    return {}


def _unpack_optional_list(payload: bytes) -> list[dict[str, Any]] | None:
    if not payload:
        return None
    loaded = json.loads(payload.decode("utf-8"))
    if loaded is None:
        return None
    if isinstance(loaded, list):
        return [dict(item) for item in loaded if isinstance(item, dict)]
    return None


def _validate_server_json_icon_fields(server_json: dict[str, Any]) -> None:
    _validate_mcp_icon_payloads(server_json.get("icons"), "server_json.icons")


def _strip_server_json_icon_response_fields(server_json: dict[str, Any]) -> dict[str, Any]:
    if "icons" not in server_json:
        return server_json
    return {
        **server_json,
        "icons": _strip_mcp_icon_response_fields(server_json.get("icons"), "server_json.icons"),
    }


def _strip_tool_icon_response_fields(tools: list[MCPTool] | None) -> list[MCPTool] | None:
    if tools is None:
        return None
    return [
        replace(
            tool,
            icons=_strip_mcp_icon_response_fields(tool.icons, f"tools[{idx}].icons"),
        )
        for idx, tool in enumerate(tools)
    ]


def _validate_tool_icons(tools: list[MCPTool] | None, field_name: str = "tools") -> None:
    if tools is None:
        return
    _validate_mcp_tool_payloads(tools, field_name)
    for idx, tool in enumerate(tools):
        _validate_mcp_icon_payloads(tool.icons, f"{field_name}[{idx}].icons")


def _resolve_mcp_server_icons(
    server_icons: list[dict[str, Any]] | None,
    version_icons: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if server_icons is not None and len(server_icons) == 0:
        return []
    resolved_icons: list[dict[str, Any]] = []
    server_icon_list = server_icons or []
    version_icon_list = version_icons or []
    server_themes: set[Any] = set()
    for icon in server_icon_list:
        if not isinstance(icon, dict):
            continue
        server_themes.add(icon.get("theme"))
        resolved_icons.append({**icon, "source": "server"})
    for icon in version_icon_list:
        if not isinstance(icon, dict) or icon.get("theme") in server_themes:
            continue
        resolved_icons.append({**icon, "source": "version"})
    return resolved_icons or None


def _validate_exactly_one(
    param1_name: str, param1_value: Any, param2_name: str, param2_value: Any
) -> None:
    if (param1_value is None) == (param2_value is None):
        raise MlflowException(
            f"Exactly one of {param1_name} or {param2_name} must be provided",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_mcp_access_endpoint_url(url: str) -> None:
    if not isinstance(url, str) or not url.strip():
        raise MlflowException(
            f"MCP access endpoint url cannot be empty or just whitespace: {url!r}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _validate_status_transition(current: MCPStatus, new: MCPStatus) -> None:
    allowed = VALID_STATUS_TRANSITIONS.get(current, set())
    if new not in allowed:
        raise MlflowException(
            f"Invalid status transition from '{current}' to '{new}'. "
            f"Allowed transitions: {sorted(str(s) for s in allowed)}",
            error_code=INVALID_PARAMETER_VALUE,
        )


def _sql_like_to_regex(pattern: str, *, case_insensitive: bool) -> re.Pattern[str]:
    parts: list[str] = ["^"]
    for char in pattern:
        if char == "%":
            parts.append(".*")
        elif char == "_":
            parts.append(".")
        else:
            parts.append(re.escape(char))
    parts.append("$")
    flags = re.DOTALL | (re.IGNORECASE if case_insensitive else 0)
    return re.compile("".join(parts), flags)


def _compare_value(actual: Any, comparator: str, expected: Any) -> bool:
    if actual is None:
        return False
    if comparator == "=":
        return bool(actual == expected)
    if comparator == "!=":
        return bool(actual != expected)
    if comparator == "<":
        return bool(actual < expected)
    if comparator == "<=":
        return bool(actual <= expected)
    if comparator == ">":
        return bool(actual > expected)
    if comparator == ">=":
        return bool(actual >= expected)
    if comparator == "LIKE":
        return (
            _sql_like_to_regex(str(expected), case_insensitive=False).match(str(actual)) is not None
        )
    if comparator == "ILIKE":
        return (
            _sql_like_to_regex(str(expected), case_insensitive=True).match(str(actual)) is not None
        )
    if comparator == "IN":
        if isinstance(expected, list | tuple | set):
            return actual in expected
        return False
    return False


class RiffDBMCPServerRegistryMixin(RiffDBGatewayAndScorerMixin):
    """Compiled RiffDB implementation of MLflow 3.15 MCP Server Registry and AI Gateway mixins."""

    def _mcp_client(self) -> Any:
        return self._application()  # type: ignore[attr-defined]

    # --- Internal MCP helpers ---

    def _fetch_mcp_server_entity(self, name: str) -> MCPServerEntity:
        client = self._mcp_client()
        res = client.get_mcpserver(GetMCPServerParams(scope=_SCOPE, name=name))
        if isinstance(res.value, GetMCPServerMissing) or not isinstance(
            res.value, GetMCPServerFound
        ):
            raise MlflowException(
                f"MCP server '{name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.server
        return MCPServerEntity(
            scope=_SCOPE,
            name=row.name,
            display_name=row.display_name,
            description=row.description,
            packed_icons=row.packed_icons,
            packed_tags=row.packed_tags,
            packed_aliases=row.packed_aliases,
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
        )

    def _fetch_mcp_server_version_entity(
        self, name: str, version: str, *, include_deleted: bool = False
    ) -> MCPServerVersionEntity:
        client = self._mcp_client()
        res = client.get_mcpserver_version(
            GetMCPServerVersionParams(scope=_SCOPE, name=name, server_version=version)
        )
        if isinstance(res.value, GetMCPServerVersionMissing) or not isinstance(
            res.value, GetMCPServerVersionFound
        ):
            raise MlflowException(
                f"MCP server version '{name}' version '{version}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.server_version
        if not include_deleted and row.status == MCPStatus.DELETED.value:
            raise MlflowException(
                f"MCP server version '{name}' version '{version}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return MCPServerVersionEntity(
            scope=_SCOPE,
            name=row.name,
            server_version=row.server_version,
            version_major=int(row.version_major),
            version_minor=int(row.version_minor),
            version_patch=int(row.version_patch),
            version_prerelease_sort_key=row.version_prerelease_sort_key,
            server_json=row.server_json,
            status=row.status,
            packed_tools=row.packed_tools,
            packed_tags=row.packed_tags,
            packed_connect_options=row.packed_connect_options,
            source_url=row.source_url,
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
        )

    def _list_mcp_server_version_entities(
        self, name: str, *, include_deleted: bool = False
    ) -> list[MCPServerVersionEntity]:
        client = self._mcp_client()
        entities: list[MCPServerVersionEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_mcpserver_versions_by_server(
                ListMCPServerVersionsByServerParams(
                    scope=_SCOPE, name=name, limit=_PAGE_BATCH_SIZE, after=cursor
                )
            )
            if not isinstance(res.value, ListMCPServerVersionsByServerFound):
                raise MlflowException("Invalid ListMCPServerVersionsByServer response from RiffDB")
            for row in res.value.versions:
                if include_deleted or row.status != MCPStatus.DELETED.value:
                    entities.append(
                        MCPServerVersionEntity(
                            scope=_SCOPE,
                            name=row.name,
                            server_version=row.server_version,
                            version_major=int(row.version_major),
                            version_minor=int(row.version_minor),
                            version_patch=int(row.version_patch),
                            version_prerelease_sort_key=row.version_prerelease_sort_key,
                            server_json=row.server_json,
                            status=row.status,
                            packed_tools=row.packed_tools,
                            packed_tags=row.packed_tags,
                            packed_connect_options=row.packed_connect_options,
                            source_url=row.source_url,
                            created_by=row.created_by,
                            last_updated_by=row.last_updated_by,
                            created_at_ms=int(row.created_at_ms),
                            last_updated_at_ms=int(row.last_updated_at_ms),
                        )
                    )
            cursor = res.next_cursor
            if not cursor or not res.value.versions:
                break
        return entities

    def _list_mcp_access_endpoint_entities(self, server_name: str) -> list[MCPAccessEndpointEntity]:
        client = self._mcp_client()
        entities: list[MCPAccessEndpointEntity] = []
        cursor: str | None = None
        while True:
            res = client.list_mcpaccess_endpoints_by_server(
                ListMCPAccessEndpointsByServerParams(
                    scope=_SCOPE,
                    server_name=server_name,
                    limit=_PAGE_BATCH_SIZE,
                    after=cursor,
                )
            )
            if not isinstance(res.value, ListMCPAccessEndpointsByServerFound):
                raise MlflowException("Invalid ListMCPAccessEndpointsByServer response from RiffDB")
            for row in res.value.endpoints:
                entities.append(
                    MCPAccessEndpointEntity(
                        scope=_SCOPE,
                        endpoint_id=row.endpoint_id,
                        server_name=row.server_name,
                        url=row.url,
                        transport_type=row.transport_type,
                        server_version=row.server_version,
                        server_alias=row.server_alias,
                        created_by=row.created_by,
                        last_updated_by=row.last_updated_by,
                        created_at_ms=int(row.created_at_ms),
                        last_updated_at_ms=int(row.last_updated_at_ms),
                    )
                )
            cursor = res.next_cursor
            if not cursor or not res.value.endpoints:
                break
        return entities

    def _resolve_latest_version_entity(self, name: str) -> MCPServerVersionEntity | None:
        versions = self._list_mcp_server_version_entities(name, include_deleted=False)
        if not versions:
            return None
        active = [v for v in versions if v.status == MCPStatus.ACTIVE.value]
        pool = active if active else versions
        return sorted(
            pool,
            key=lambda v: (
                int(v.version_major),
                int(v.version_minor),
                int(v.version_patch),
                v.version_prerelease_sort_key,
                int(v.created_at_ms),
                v.server_version,
            ),
            reverse=True,
        )[0]

    def _version_entity_to_mlflow(
        self,
        entity: MCPServerVersionEntity,
        aliases_map: dict[str, str] | None = None,
    ) -> MCPServerVersion:
        if aliases_map is None:
            try:
                server_entity = self._fetch_mcp_server_entity(entity.name)
                aliases_map = {
                    str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
                }
            except MlflowException:
                aliases_map = {}
        alias_names = [alias for alias, ver in aliases_map.items() if ver == entity.server_version]
        raw_tools = _unpack_optional_list(entity.packed_tools)
        tools = None if raw_tools is None else [MCPTool.from_dict(t) for t in raw_tools]
        tags = {str(k): str(v) for k, v in _unpack_dict(entity.packed_tags).items()}
        raw_connect_options = _unpack_dict(entity.packed_connect_options)
        connect_options = {
            str(k): ConnectOptionSettings(**v)
            for k, v in raw_connect_options.items()
            if isinstance(v, dict)
        }
        server_json = json.loads(entity.server_json) if entity.server_json else {}
        return MCPServerVersion(
            name=entity.name,
            version=entity.server_version,
            server_json=server_json,
            status=MCPStatus(entity.status),
            tools=tools,
            aliases=alias_names,
            tags=tags,
            connect_options=connect_options,
            source=entity.source_url,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=int(entity.created_at_ms),
            last_updated_timestamp=int(entity.last_updated_at_ms),
        )

    def _resolve_endpoint_version(
        self,
        endpoint: MCPAccessEndpointEntity,
        *,
        server_entity: MCPServerEntity | None = None,
    ) -> MCPServerVersion | None:
        if server_entity is None:
            try:
                server_entity = self._fetch_mcp_server_entity(endpoint.server_name)
            except MlflowException:
                return None
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        if endpoint.server_version is not None:
            try:
                sv = self._fetch_mcp_server_version_entity(
                    endpoint.server_name, endpoint.server_version, include_deleted=False
                )
                return self._version_entity_to_mlflow(sv, aliases_map)
            except MlflowException:
                return None
        if endpoint.server_alias == "latest":
            sv_opt = self._resolve_latest_version_entity(endpoint.server_name)
            if sv_opt is None:
                return None
            return self._version_entity_to_mlflow(sv_opt, aliases_map)
        if endpoint.server_alias is not None:
            target_version = aliases_map.get(endpoint.server_alias)
            if target_version is None:
                return None
            try:
                sv = self._fetch_mcp_server_version_entity(
                    endpoint.server_name, target_version, include_deleted=False
                )
                return self._version_entity_to_mlflow(sv, aliases_map)
            except MlflowException:
                return None
        return None

    def _endpoint_entity_to_mlflow(
        self,
        endpoint: MCPAccessEndpointEntity,
        resolved_version: MCPServerVersion,
    ) -> MCPAccessEndpoint:
        return MCPAccessEndpoint(
            id=endpoint.endpoint_id,
            server_name=endpoint.server_name,
            url=endpoint.url,
            transport_type=MCPRemoteTransportType(endpoint.transport_type),
            server_version=endpoint.server_version,
            server_alias=endpoint.server_alias,
            resolved_version=resolved_version,
            created_by=endpoint.created_by,
            last_updated_by=endpoint.last_updated_by,
            creation_timestamp=int(endpoint.created_at_ms),
            last_updated_timestamp=int(endpoint.last_updated_at_ms),
        )

    def _server_entity_to_mlflow(self, entity: MCPServerEntity) -> MCPServer:
        tags = {str(k): str(v) for k, v in _unpack_dict(entity.packed_tags).items()}
        aliases = {str(k): str(v) for k, v in _unpack_dict(entity.packed_aliases).items()}
        latest_sv = self._resolve_latest_version_entity(entity.name)
        resolved_latest_version = latest_sv.server_version if latest_sv is not None else None
        status = MCPStatus(latest_sv.status) if latest_sv is not None else None
        parent_server_json = (
            json.loads(latest_sv.server_json)
            if latest_sv is not None and latest_sv.server_json
            else None
        )
        description = entity.description
        if description is None and isinstance(parent_server_json, dict):
            description = parent_server_json.get("description")
        server_icons = _unpack_optional_list(entity.packed_icons)
        version_icons = (
            parent_server_json.get("icons") if isinstance(parent_server_json, dict) else None
        )
        icons = _resolve_mcp_server_icons(server_icons, version_icons)
        endpoint_entities: list[MCPAccessEndpoint] = []
        for ep in self._list_mcp_access_endpoint_entities(entity.name):
            resolved = self._resolve_endpoint_version(ep, server_entity=entity)
            if resolved is not None:
                endpoint_entities.append(self._endpoint_entity_to_mlflow(ep, resolved))
        return MCPServer(
            name=entity.name,
            display_name=entity.display_name,
            description=description,
            icons=icons,
            status=status,
            tags=tags,
            aliases=aliases,
            access_endpoints=endpoint_entities,
            latest_version=resolved_latest_version,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=int(entity.created_at_ms),
            last_updated_timestamp=int(entity.last_updated_at_ms),
        )

    def _save_mcp_server_entity(self, entity: MCPServerEntity) -> MCPServerEntity:
        client = self._mcp_client()
        res = client.upsert_mcpserver_entity(
            UpsertMCPServerEntityInput(
                request_id=str(uuid.uuid4()),
                name=entity.name,
                display_name=entity.display_name,
                description=entity.description,
                packed_icons=entity.packed_icons,
                packed_tags=entity.packed_tags,
                packed_aliases=entity.packed_aliases,
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
            )
        )
        if not isinstance(res.outcome, UpsertMCPServerEntityMCPServerUpserted):
            raise MlflowException("Invalid UpsertMCPServerEntity response from RiffDB")
        return res.outcome.server

    def _save_mcp_server_version_entity(
        self, entity: MCPServerVersionEntity
    ) -> MCPServerVersionEntity:
        client = self._mcp_client()
        res = client.upsert_mcpserver_version_entity(
            UpsertMCPServerVersionEntityInput(
                request_id=str(uuid.uuid4()),
                name=entity.name,
                server_version=entity.server_version,
                version_major=int(entity.version_major),
                version_minor=int(entity.version_minor),
                version_patch=int(entity.version_patch),
                version_prerelease_sort_key=entity.version_prerelease_sort_key,
                server_json=entity.server_json,
                status=entity.status,
                packed_tools=entity.packed_tools,
                packed_tags=entity.packed_tags,
                packed_connect_options=entity.packed_connect_options,
                source_url=entity.source_url,
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
                created_at_ms=int(entity.created_at_ms),
                last_updated_at_ms=int(entity.last_updated_at_ms),
            )
        )
        if not isinstance(res.outcome, UpsertMCPServerVersionEntityMCPServerVersionUpserted):
            raise MlflowException("Invalid UpsertMCPServerVersionEntity response from RiffDB")
        return res.outcome.server_version

    def _delete_latest_alias_endpoints_if_unresolvable(self, server_name: str) -> None:
        if self._resolve_latest_version_entity(server_name) is not None:
            return
        client = self._mcp_client()
        for ep in self._list_mcp_access_endpoint_entities(server_name):
            if ep.server_alias == "latest":
                client.delete_mcpaccess_endpoint_entity(
                    DeleteMCPAccessEndpointEntityInput(
                        request_id=str(uuid.uuid4()),
                        endpoint_id=ep.endpoint_id,
                    )
                )

    # --- MCPServer operations ---

    def create_mcp_server(
        self,
        name: str,
        description: str | None = None,
        icons: list[MCPIcon] | None = None,
        created_by: str | None = None,
    ) -> MCPServer:
        validate_mcp_server_name(name)
        cleaned_icons = _strip_mcp_icon_response_fields(icons)  # type: ignore[arg-type]
        _validate_mcp_icon_payloads(cleaned_icons, "icons")
        now = _now_ms()
        client = self._mcp_client()
        res = client.create_mcpserver_entity(
            CreateMCPServerEntityInput(
                request_id=str(uuid.uuid4()),
                name=name,
                display_name=None,
                description=description,
                packed_icons=_pack_json(cleaned_icons) if cleaned_icons is not None else b"",
                packed_tags=_pack_json({}),
                packed_aliases=_pack_json({}),
                created_by=created_by,
                last_updated_by=created_by,
                created_at_ms=now,
                last_updated_at_ms=now,
            )
        )
        if isinstance(res.outcome, CreateMCPServerEntityMCPServerExists):
            raise MlflowException(
                f"MCP server with name '{name}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )
        if not isinstance(res.outcome, CreateMCPServerEntityMCPServerCreated):
            raise MlflowException("Invalid CreateMCPServerEntity response from RiffDB")
        return self._server_entity_to_mlflow(res.outcome.server)

    def get_mcp_server(self, name: str) -> MCPServer:
        entity = self._fetch_mcp_server_entity(name)
        return self._server_entity_to_mlflow(entity)

    def search_mcp_servers(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServer]:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value for max_results. It must be a positive integer, got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MCP_SERVER_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value for max_results. It must be at most "
                f"{SEARCH_MCP_SERVER_MAX_RESULTS_THRESHOLD}, but got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )
        client = self._mcp_client()
        servers: list[MCPServer] = []
        cursor: str | None = None
        while True:
            res = client.search_mcpservers(
                SearchMCPServersParams(scope=_SCOPE, limit=_PAGE_BATCH_SIZE, after=cursor)
            )
            if not isinstance(res.value, SearchMCPServersFound):
                raise MlflowException("Invalid SearchMCPServers response from RiffDB")
            for row in res.value.servers:
                entity = MCPServerEntity(
                    scope=_SCOPE,
                    name=row.name,
                    display_name=row.display_name,
                    description=row.description,
                    packed_icons=row.packed_icons,
                    packed_tags=row.packed_tags,
                    packed_aliases=row.packed_aliases,
                    created_by=row.created_by,
                    last_updated_by=row.last_updated_by,
                    created_at_ms=int(row.created_at_ms),
                    last_updated_at_ms=int(row.last_updated_at_ms),
                )
                servers.append(self._server_entity_to_mlflow(entity))
            cursor = res.next_cursor
            if not cursor or not res.value.servers:
                break
        if filter_string:
            servers = self._filter_mcp_servers(servers, filter_string)
        servers = self._sort_mcp_servers(servers, order_by)
        window = servers[offset : offset + max_results + 1]
        next_token = None
        if len(window) > max_results:
            next_token = SearchUtils.create_page_token(  # type: ignore[no-untyped-call]
                offset + max_results
            )
        return PagedList(window[:max_results], next_token)

    def _filter_mcp_servers(self, servers: list[MCPServer], filter_string: str) -> list[MCPServer]:
        parsed = SearchMCPServerUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
        filtered = list(servers)
        for clause in parsed:
            type_ = clause["type"]
            key = clause["key"]
            comparator = clause["comparator"]
            value = clause["value"]
            if comparator not in _VALID_FILTER_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' for {type_} '{key}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if type_ == "attribute":
                if key == "status":
                    filtered = [
                        s
                        for s in filtered
                        if _compare_value(
                            s.status.value if s.status is not None else None,
                            comparator,
                            value,
                        )
                    ]
                elif key == "has_access_endpoints":
                    if comparator != "=" or str(value).lower() not in ("true", "false"):
                        raise MlflowException(
                            "has_access_endpoints only supports '= true' or '= false'",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    want_endpoints = str(value).lower() == "true"
                    filtered = [s for s in filtered if bool(s.access_endpoints) == want_endpoints]
                else:
                    attr_map = {
                        "name": lambda s: s.name,
                        "display_name": lambda s: s.display_name,
                        "description": lambda s: s.description,
                        "created_by": lambda s: s.created_by,
                        "last_updated_by": lambda s: s.last_updated_by,
                        "created_at": lambda s: s.creation_timestamp,
                        "last_updated_at": lambda s: s.last_updated_timestamp,
                    }
                    getter = attr_map.get(key)
                    if getter is None:
                        raise MlflowException(
                            f"Invalid attribute '{key}' in MCP server filter.",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    filtered = [s for s in filtered if _compare_value(getter(s), comparator, value)]
            elif type_ == "tag":
                filtered = [
                    s
                    for s in filtered
                    if key in s.tags and _compare_value(s.tags[key], comparator, value)
                ]
            else:
                raise MlflowException(
                    f"Invalid filter type '{type_}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        return filtered

    def _sort_mcp_servers(
        self, servers: list[MCPServer], order_by_list: list[str] | None
    ) -> list[MCPServer]:
        valid_keys = {"name", "created_at", "last_updated_at"}
        clauses: list[tuple[str, bool]] = []
        observed: set[str] = set()
        if order_by_list:
            for order_by_clause in order_by_list:
                token_value, is_ascending = SearchUtils._parse_order_by_string(  # type: ignore[no-untyped-call]
                    order_by_clause
                )
                key = token_value.strip()
                if key not in valid_keys:
                    raise MlflowException(
                        f"Invalid order_by key '{key}'. Valid keys: {sorted(valid_keys)}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if key in observed:
                    raise MlflowException(
                        f"Duplicate order_by field: '{key}'",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                observed.add(key)
                clauses.append((key, is_ascending))
        if "name" not in observed:
            clauses.append(("name", True))
        result = list(servers)
        for key, is_ascending in reversed(clauses):
            if key == "name":
                result.sort(key=lambda s: s.name, reverse=not is_ascending)
            elif key == "created_at":
                result.sort(key=lambda s: s.creation_timestamp or 0, reverse=not is_ascending)
            elif key == "last_updated_at":
                result.sort(key=lambda s: s.last_updated_timestamp or 0, reverse=not is_ascending)
        return result

    def update_mcp_server(
        self,
        name: str,
        description: str | None = _NOT_SET,
        display_name: str | None = _NOT_SET,
        icons: list[MCPIcon] | None = _NOT_SET,
        last_updated_by: str | None = None,
    ) -> MCPServer:
        packed_icons_override: bytes | None = None
        if icons is not NOT_SET:
            cleaned_icons = _strip_mcp_icon_response_fields(icons)  # type: ignore[arg-type]
            _validate_mcp_icon_payloads(cleaned_icons, "icons")
            packed_icons_override = _pack_json(cleaned_icons) if cleaned_icons is not None else b""
        entity = self._fetch_mcp_server_entity(name)
        updated = MCPServerEntity(
            scope=_SCOPE,
            name=entity.name,
            display_name=entity.display_name if display_name is NOT_SET else display_name,
            description=entity.description if description is NOT_SET else description,
            packed_icons=(
                entity.packed_icons if packed_icons_override is None else packed_icons_override
            ),
            packed_tags=entity.packed_tags,
            packed_aliases=entity.packed_aliases,
            created_by=entity.created_by,
            last_updated_by=last_updated_by,
            created_at_ms=entity.created_at_ms,
            last_updated_at_ms=_now_ms(),
        )
        saved = self._save_mcp_server_entity(updated)
        return self._server_entity_to_mlflow(saved)

    def delete_mcp_server(self, name: str) -> None:
        self._fetch_mcp_server_entity(name)
        versions = self._list_mcp_server_version_entities(name, include_deleted=False)
        for v in versions:
            if v.status == MCPStatus.ACTIVE.value:
                raise MlflowException(
                    f"Cannot delete MCP server '{name}' while it still has an active version "
                    f"('{v.server_version}'). Delete or deactivate the active version first.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        client = self._mcp_client()
        for ep in self._list_mcp_access_endpoint_entities(name):
            client.delete_mcpaccess_endpoint_entity(
                DeleteMCPAccessEndpointEntityInput(
                    request_id=str(uuid.uuid4()),
                    endpoint_id=ep.endpoint_id,
                )
            )
        res = client.delete_mcpserver_entity(
            DeleteMCPServerEntityInput(
                request_id=str(uuid.uuid4()),
                name=name,
            )
        )
        if isinstance(res.outcome, DeleteMCPServerEntityMCPServerMissing):
            raise MlflowException(
                f"MCP server '{name}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(res.outcome, DeleteMCPServerEntityMCPServerDeleted):
            raise MlflowException("Invalid DeleteMCPServerEntity response from RiffDB")

    # --- MCPServerVersion operations ---

    def create_mcp_server_version(
        self,
        server_json: dict[str, Any],
        source: str | None = None,
        status: MCPStatus | None = None,
        tools: list[MCPTool] | None = _NOT_SET,
        created_by: str | None = None,
        connect_options: dict[str, ConnectOptionSettings] | None = None,
    ) -> MCPServerVersion:
        name = server_json.get("name")
        version = server_json.get("version")
        if not name or not version:
            raise MlflowException(
                "server_json must contain 'name' and 'version' keys",
                error_code=INVALID_PARAMETER_VALUE,
            )
        validate_mcp_server_name(name)
        cleaned_server_json = _strip_server_json_icon_response_fields(server_json)
        _validate_server_json_icon_fields(cleaned_server_json)
        parsed_version = parse_semver(version, param_name="server_json.version")

        now = _now_ms()
        resolved_status = status or MCPStatus.DRAFT
        _validate_mcp_initial_status(resolved_status)

        cleaned_tools = None if tools is NOT_SET else _strip_tool_icon_response_fields(tools)
        _validate_tool_icons(cleaned_tools)
        tools_json = None if cleaned_tools is None else [t.to_dict() for t in cleaned_tools]
        connect_options_dict = (
            {k: asdict(v) for k, v in connect_options.items()}
            if connect_options is not None
            else {}
        )

        client = self._mcp_client()
        try:
            server_entity = self._fetch_mcp_server_entity(name)
        except MlflowException:
            create_res = client.create_mcpserver_entity(
                CreateMCPServerEntityInput(
                    request_id=str(uuid.uuid4()),
                    name=name,
                    display_name=None,
                    description=None,
                    packed_icons=b"",
                    packed_tags=_pack_json({}),
                    packed_aliases=_pack_json({}),
                    created_by=created_by,
                    last_updated_by=created_by,
                    created_at_ms=now,
                    last_updated_at_ms=now,
                )
            )
            if isinstance(create_res.outcome, CreateMCPServerEntityMCPServerCreated):
                server_entity = create_res.outcome.server
            else:
                server_entity = self._fetch_mcp_server_entity(name)

        res = client.create_mcpserver_version_entity(
            CreateMCPServerVersionEntityInput(
                request_id=str(uuid.uuid4()),
                name=name,
                server_version=version,
                version_major=int(parsed_version.major),
                version_minor=int(parsed_version.minor),
                version_patch=int(parsed_version.patch),
                version_prerelease_sort_key=encode_prerelease_sort_key(parsed_version),
                server_json=json.dumps(cleaned_server_json, separators=(",", ":"), sort_keys=True),
                status=resolved_status.value,
                packed_tools=_pack_json(tools_json) if tools_json is not None else b"",
                packed_tags=_pack_json({}),
                packed_connect_options=_pack_json(connect_options_dict),
                source_url=source,
                created_by=created_by,
                last_updated_by=created_by,
                created_at_ms=now,
                last_updated_at_ms=now,
            )
        )
        if isinstance(res.outcome, CreateMCPServerVersionEntityMCPServerVersionExists):
            raise MlflowException(
                f"MCP server version '{name}' version '{version}' already exists",
                error_code=RESOURCE_ALREADY_EXISTS,
            )
        if not isinstance(res.outcome, CreateMCPServerVersionEntityMCPServerVersionCreated):
            raise MlflowException("Invalid CreateMCPServerVersionEntity response from RiffDB")
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        return self._version_entity_to_mlflow(res.outcome.server_version, aliases_map)

    def get_mcp_server_version(self, name: str, version: str) -> MCPServerVersion:
        sv = self._fetch_mcp_server_version_entity(name, version, include_deleted=False)
        return self._version_entity_to_mlflow(sv)

    def get_mcp_server_version_by_alias(self, name: str, alias: str) -> MCPServerVersion:
        if alias == "latest":
            return self.get_latest_mcp_server_version(name)
        server_entity = self._fetch_mcp_server_entity(name)
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        target_version = aliases_map.get(alias)
        if target_version is None:
            raise MlflowException(
                f"Alias '{alias}' not found for MCP server '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return self.get_mcp_server_version(name, target_version)

    def get_latest_mcp_server_version(self, name: str) -> MCPServerVersion:
        server_entity = self._fetch_mcp_server_entity(name)
        sv = self._resolve_latest_version_entity(name)
        if sv is None:
            raise MlflowException(
                f"No resolved latest version found for MCP server '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        return self._version_entity_to_mlflow(sv, aliases_map)

    def search_mcp_server_versions(
        self,
        name: str,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServerVersion]:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value for max_results. It must be a positive integer, got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )
        try:
            server_entity = self._fetch_mcp_server_entity(name)
            aliases_map = {
                str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
            }
        except MlflowException:
            aliases_map = {}
        entities = self._list_mcp_server_version_entities(name, include_deleted=False)
        if filter_string:
            entities = self._filter_mcp_server_versions(entities, filter_string)
        entities = self._sort_mcp_server_versions(entities, order_by)
        versions = [self._version_entity_to_mlflow(e, aliases_map) for e in entities]
        window = versions[offset : offset + max_results + 1]
        next_token = None
        if len(window) > max_results:
            next_token = SearchUtils.create_page_token(  # type: ignore[no-untyped-call]
                offset + max_results
            )
        return PagedList(window[:max_results], next_token)

    def _filter_mcp_server_versions(
        self, entities: list[MCPServerVersionEntity], filter_string: str
    ) -> list[MCPServerVersionEntity]:
        normalized = re.sub(
            r"(?<![`\w.])version(?=\s*(?:=|!=|<=|>=|<|>|LIKE|ILIKE))",
            "`version`",
            filter_string,
        )
        parsed = SearchMCPServerVersionUtils.parse_search_filter(normalized)  # type: ignore[no-untyped-call]
        filtered = list(entities)
        for clause in parsed:
            type_ = clause["type"]
            key = clause["key"]
            comparator = clause["comparator"]
            value = clause["value"]
            if type_ != "attribute":
                raise MlflowException(
                    f"Invalid filter type '{type_}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if comparator not in _VALID_FILTER_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' for attribute '{key}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if key == "version":
                if comparator not in {"=", "!=", "<", "<=", ">", ">="}:
                    raise MlflowException.invalid_parameter_value(
                        "version only supports semantic comparators "
                        "'=', '!=', '<', '<=', '>', and '>='"
                    )
                if comparator == "=":
                    filtered = [e for e in filtered if e.server_version == value]
                elif comparator == "!=":
                    filtered = [e for e in filtered if e.server_version != value]
                else:
                    parsed_ver = parse_semver(value, param_name="filter_string version")
                    target = (
                        int(parsed_ver.major),
                        int(parsed_ver.minor),
                        int(parsed_ver.patch),
                        encode_prerelease_sort_key(parsed_ver),
                    )
                    filtered = [
                        e
                        for e in filtered
                        if _compare_value(
                            (
                                int(e.version_major),
                                int(e.version_minor),
                                int(e.version_patch),
                                e.version_prerelease_sort_key,
                            ),
                            comparator,
                            target,
                        )
                    ]
            else:
                attr_map = {
                    "status": lambda e: e.status,
                    "source": lambda e: e.source_url,
                    "created_by": lambda e: e.created_by,
                    "last_updated_by": lambda e: e.last_updated_by,
                    "created_at": lambda e: e.created_at_ms,
                    "last_updated_at": lambda e: e.last_updated_at_ms,
                }
                getter = attr_map.get(key)
                if getter is None:
                    raise MlflowException(
                        f"Invalid attribute '{key}' in MCP server version filter.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                filtered = [e for e in filtered if _compare_value(getter(e), comparator, value)]
        return filtered

    def _sort_mcp_server_versions(
        self, entities: list[MCPServerVersionEntity], order_by_list: list[str] | None
    ) -> list[MCPServerVersionEntity]:
        valid_keys = {"version", "created_at", "last_updated_at"}
        clauses: list[tuple[str, bool]] = []
        observed: set[str] = set()
        if order_by_list:
            for order_by_clause in order_by_list:
                token_value, is_ascending = SearchUtils._parse_order_by_string(  # type: ignore[no-untyped-call]
                    order_by_clause
                )
                key = token_value.strip()
                if key not in valid_keys:
                    raise MlflowException(
                        f"Invalid order_by key '{key}'. Valid keys: {sorted(valid_keys)}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if key in observed:
                    raise MlflowException(
                        f"Duplicate order_by field: '{key}'",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                observed.add(key)
                clauses.append((key, is_ascending))
        if "created_at" not in observed:
            clauses.append(("created_at", True))
        result = sorted(entities, key=lambda e: e.server_version)
        for key, is_ascending in reversed(clauses):
            if key == "version":
                result.sort(
                    key=lambda e: (
                        int(e.version_major),
                        int(e.version_minor),
                        int(e.version_patch),
                        e.version_prerelease_sort_key,
                    ),
                    reverse=not is_ascending,
                )
            elif key == "created_at":
                result.sort(key=lambda e: int(e.created_at_ms), reverse=not is_ascending)
            elif key == "last_updated_at":
                result.sort(key=lambda e: int(e.last_updated_at_ms), reverse=not is_ascending)
        return result

    def update_mcp_server_version(
        self,
        name: str,
        version: str,
        status: MCPStatus | None = _NOT_SET,
        tools: list[MCPTool] | None = _NOT_SET,
        last_updated_by: str | None = None,
        connect_options: dict[str, ConnectOptionSettings] | None = _NOT_SET,
    ) -> MCPServerVersion:
        packed_tools_override: bytes | None = None
        if tools is not NOT_SET:
            cleaned_tools = _strip_tool_icon_response_fields(tools)
            _validate_tool_icons(cleaned_tools)
            tools_json = None if cleaned_tools is None else [t.to_dict() for t in cleaned_tools]
            packed_tools_override = _pack_json(tools_json) if tools_json is not None else b""

        sv = self._fetch_mcp_server_version_entity(name, version, include_deleted=False)
        if status is None:
            raise MlflowException.invalid_parameter_value(
                "status cannot be null; omit the field to leave it unchanged"
            )
        new_status = sv.status
        if status is not NOT_SET:
            _validate_status_transition(MCPStatus(sv.status), status)
            new_status = status.value

        packed_connect_override = sv.packed_connect_options
        if connect_options is not NOT_SET:
            connect_dict = (
                {k: asdict(v) for k, v in connect_options.items()}
                if connect_options is not None
                else {}
            )
            packed_connect_override = _pack_json(connect_dict)

        updated = MCPServerVersionEntity(
            scope=_SCOPE,
            name=sv.name,
            server_version=sv.server_version,
            version_major=sv.version_major,
            version_minor=sv.version_minor,
            version_patch=sv.version_patch,
            version_prerelease_sort_key=sv.version_prerelease_sort_key,
            server_json=sv.server_json,
            status=new_status,
            packed_tools=(
                sv.packed_tools if packed_tools_override is None else packed_tools_override
            ),
            packed_tags=sv.packed_tags,
            packed_connect_options=packed_connect_override,
            source_url=sv.source_url,
            created_by=sv.created_by,
            last_updated_by=last_updated_by,
            created_at_ms=sv.created_at_ms,
            last_updated_at_ms=_now_ms(),
        )
        saved = self._save_mcp_server_version_entity(updated)
        if status is not NOT_SET:
            self._delete_latest_alias_endpoints_if_unresolvable(name)
        return self._version_entity_to_mlflow(saved)

    def delete_mcp_server_version(self, name: str, version: str) -> None:
        sv = self._fetch_mcp_server_version_entity(name, version, include_deleted=True)
        _validate_status_transition(MCPStatus(sv.status), MCPStatus.DELETED)
        server_entity = self._fetch_mcp_server_entity(name)
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        removed_aliases = [alias for alias, ver in aliases_map.items() if ver == version]
        if removed_aliases:
            for alias in removed_aliases:
                aliases_map.pop(alias, None)
            self._save_mcp_server_entity(
                MCPServerEntity(
                    scope=_SCOPE,
                    name=server_entity.name,
                    display_name=server_entity.display_name,
                    description=server_entity.description,
                    packed_icons=server_entity.packed_icons,
                    packed_tags=server_entity.packed_tags,
                    packed_aliases=_pack_json(aliases_map),
                    created_by=server_entity.created_by,
                    last_updated_by=server_entity.last_updated_by,
                    created_at_ms=server_entity.created_at_ms,
                    last_updated_at_ms=_now_ms(),
                )
            )
        client = self._mcp_client()
        for ep in self._list_mcp_access_endpoint_entities(name):
            if ep.server_version == version or (
                ep.server_alias is not None and ep.server_alias in removed_aliases
            ):
                client.delete_mcpaccess_endpoint_entity(
                    DeleteMCPAccessEndpointEntityInput(
                        request_id=str(uuid.uuid4()),
                        endpoint_id=ep.endpoint_id,
                    )
                )
        self._save_mcp_server_version_entity(
            MCPServerVersionEntity(
                scope=_SCOPE,
                name=sv.name,
                server_version=sv.server_version,
                version_major=sv.version_major,
                version_minor=sv.version_minor,
                version_patch=sv.version_patch,
                version_prerelease_sort_key=sv.version_prerelease_sort_key,
                server_json=sv.server_json,
                status=MCPStatus.DELETED.value,
                packed_tools=sv.packed_tools,
                packed_tags=sv.packed_tags,
                packed_connect_options=sv.packed_connect_options,
                source_url=sv.source_url,
                created_by=sv.created_by,
                last_updated_by=sv.last_updated_by,
                created_at_ms=sv.created_at_ms,
                last_updated_at_ms=_now_ms(),
            )
        )
        self._delete_latest_alias_endpoints_if_unresolvable(name)

    # --- MCPAccessEndpoint operations ---

    def _validate_alias_target_version(
        self, server_entity: MCPServerEntity, server_alias: str
    ) -> MCPServerVersionEntity:
        if server_alias == "latest":
            latest = self._resolve_latest_version_entity(server_entity.name)
            if latest is None:
                raise MlflowException(
                    f"No resolved latest version found for MCP server '{server_entity.name}'",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            return latest
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        target_version = aliases_map.get(server_alias)
        if target_version is None:
            raise MlflowException(
                f"Alias '{server_alias}' not found for MCP server '{server_entity.name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        sv = self._fetch_mcp_server_version_entity(
            server_entity.name, target_version, include_deleted=True
        )
        if sv.status == MCPStatus.DELETED.value:
            raise MlflowException(
                f"Alias '{server_alias}' for MCP server '{server_entity.name}' "
                f"points to deleted version '{target_version}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return sv

    def create_mcp_access_endpoint(
        self,
        server_name: str,
        url: str,
        transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
        server_version: str | None = None,
        server_alias: str | None = None,
        created_by: str | None = None,
    ) -> MCPAccessEndpoint:
        _validate_exactly_one("server_version", server_version, "server_alias", server_alias)
        _validate_mcp_access_endpoint_url(url)
        server_entity = self._fetch_mcp_server_entity(server_name)
        if server_version is not None:
            sv = self._fetch_mcp_server_version_entity(
                server_name, server_version, include_deleted=True
            )
            if sv.status == MCPStatus.DELETED.value:
                raise MlflowException(
                    f"Cannot create MCP access endpoint to deleted "
                    f"MCP server version '{server_name}' version '{server_version}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        if server_alias is not None:
            self._validate_alias_target_version(server_entity, server_alias)

        now = _now_ms()
        endpoint_id = f"ae-{uuid.uuid4().hex}"
        client = self._mcp_client()
        res = client.upsert_mcpaccess_endpoint_entity(
            UpsertMCPAccessEndpointEntityInput(
                request_id=str(uuid.uuid4()),
                endpoint_id=endpoint_id,
                server_name=server_name,
                url=url,
                transport_type=transport_type.value,
                server_version=server_version,
                server_alias=server_alias,
                created_by=created_by,
                last_updated_by=created_by,
                created_at_ms=now,
                last_updated_at_ms=now,
            )
        )
        if not isinstance(res.outcome, UpsertMCPAccessEndpointEntityMCPAccessEndpointUpserted):
            raise MlflowException("Invalid UpsertMCPAccessEndpointEntity response from RiffDB")
        resolved = self._resolve_endpoint_version(res.outcome.endpoint, server_entity=server_entity)
        assert resolved is not None
        return self._endpoint_entity_to_mlflow(res.outcome.endpoint, resolved)

    def get_mcp_access_endpoint(self, server_name: str, endpoint_id: str) -> MCPAccessEndpoint:
        client = self._mcp_client()
        res = client.get_mcpaccess_endpoint(
            GetMCPAccessEndpointParams(scope=_SCOPE, endpoint_id=endpoint_id)
        )
        if isinstance(res.value, GetMCPAccessEndpointMissing) or not isinstance(
            res.value, GetMCPAccessEndpointFound
        ):
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        row = res.value.endpoint
        if row.server_name != server_name:
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} does not belong to server '{server_name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ep = MCPAccessEndpointEntity(
            scope=_SCOPE,
            endpoint_id=row.endpoint_id,
            server_name=row.server_name,
            url=row.url,
            transport_type=row.transport_type,
            server_version=row.server_version,
            server_alias=row.server_alias,
            created_by=row.created_by,
            last_updated_by=row.last_updated_by,
            created_at_ms=int(row.created_at_ms),
            last_updated_at_ms=int(row.last_updated_at_ms),
        )
        resolved = self._resolve_endpoint_version(ep)
        if resolved is None:
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return self._endpoint_entity_to_mlflow(ep, resolved)

    def search_mcp_access_endpoints(
        self,
        server_name: str | None = None,
        server_version: str | None = None,
        server_alias: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPAccessEndpoint]:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                f"Invalid value for max_results. It must be a positive integer, got {max_results}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        offset = SearchUtils.parse_start_offset_from_page_token(  # type: ignore[no-untyped-call]
            page_token
        )
        if server_name is not None:
            raw_endpoints = self._list_mcp_access_endpoint_entities(server_name)
        else:
            client = self._mcp_client()
            raw_endpoints = []
            cursor: str | None = None
            while True:
                res = client.search_mcpaccess_endpoints(
                    SearchMCPAccessEndpointsParams(
                        scope=_SCOPE, limit=_PAGE_BATCH_SIZE, after=cursor
                    )
                )
                if not isinstance(res.value, SearchMCPAccessEndpointsFound):
                    raise MlflowException("Invalid SearchMCPAccessEndpoints response from RiffDB")
                for row in res.value.endpoints:
                    raw_endpoints.append(
                        MCPAccessEndpointEntity(
                            scope=_SCOPE,
                            endpoint_id=row.endpoint_id,
                            server_name=row.server_name,
                            url=row.url,
                            transport_type=row.transport_type,
                            server_version=row.server_version,
                            server_alias=row.server_alias,
                            created_by=row.created_by,
                            last_updated_by=row.last_updated_by,
                            created_at_ms=int(row.created_at_ms),
                            last_updated_at_ms=int(row.last_updated_at_ms),
                        )
                    )
                cursor = res.next_cursor
                if not cursor or not res.value.endpoints:
                    break
        endpoints: list[MCPAccessEndpoint] = []
        for ep in raw_endpoints:
            if server_version is not None and ep.server_version != server_version:
                continue
            if server_alias is not None and ep.server_alias != server_alias:
                continue
            resolved = self._resolve_endpoint_version(ep)
            if resolved is not None:
                endpoints.append(self._endpoint_entity_to_mlflow(ep, resolved))
        if filter_string:
            endpoints = self._filter_mcp_access_endpoints(endpoints, filter_string)
        endpoints = self._sort_mcp_access_endpoints(endpoints, order_by)
        window = endpoints[offset : offset + max_results + 1]
        next_token = None
        if len(window) > max_results:
            next_token = SearchUtils.create_page_token(  # type: ignore[no-untyped-call]
                offset + max_results
            )
        return PagedList(window[:max_results], next_token)

    def _filter_mcp_access_endpoints(
        self, endpoints: list[MCPAccessEndpoint], filter_string: str
    ) -> list[MCPAccessEndpoint]:
        parsed = SearchMCPAccessEndpointUtils.parse_search_filter(filter_string)  # type: ignore[no-untyped-call]
        filtered = list(endpoints)
        for clause in parsed:
            type_ = clause["type"]
            key = clause["key"]
            comparator = clause["comparator"]
            value = clause["value"]
            if type_ != "attribute":
                raise MlflowException(
                    f"Invalid filter type '{type_}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if comparator not in _VALID_FILTER_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' for attribute '{key}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            attr_map: dict[str, Callable[[MCPAccessEndpoint], Any]] = {
                "id": lambda ep: ep.id,
                "server_name": lambda ep: ep.server_name,
                "server_version": lambda ep: ep.server_version,
                "server_alias": lambda ep: ep.server_alias,
                "url": lambda ep: ep.url,
                "transport_type": lambda ep: ep.transport_type.value,
                "status": lambda ep: (
                    ep.resolved_version.status.value if ep.resolved_version is not None else None
                ),
                "created_by": lambda ep: ep.created_by,
                "last_updated_by": lambda ep: ep.last_updated_by,
                "created_at": lambda ep: ep.creation_timestamp,
                "last_updated_at": lambda ep: ep.last_updated_timestamp,
            }
            getter = attr_map.get(key)
            if getter is None:
                raise MlflowException(
                    f"Invalid attribute '{key}' in MCP access endpoint filter.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            filtered = [ep for ep in filtered if _compare_value(getter(ep), comparator, value)]
        return filtered

    def _sort_mcp_access_endpoints(
        self, endpoints: list[MCPAccessEndpoint], order_by_list: list[str] | None
    ) -> list[MCPAccessEndpoint]:
        valid_keys = {"id", "server_name", "created_at", "last_updated_at"}
        clauses: list[tuple[str, bool]] = []
        observed: set[str] = set()
        if order_by_list:
            for order_by_clause in order_by_list:
                token_value, is_ascending = SearchUtils._parse_order_by_string(  # type: ignore[no-untyped-call]
                    order_by_clause
                )
                key = token_value.strip()
                if key not in valid_keys:
                    raise MlflowException(
                        f"Invalid order_by key '{key}'. Valid keys: {sorted(valid_keys)}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if key in observed:
                    raise MlflowException(
                        f"Duplicate order_by field: '{key}'",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                observed.add(key)
                clauses.append((key, is_ascending))
        if "id" not in observed:
            clauses.append(("id", True))
        result = list(endpoints)
        for key, is_ascending in reversed(clauses):
            if key == "id":
                result.sort(key=lambda ep: ep.id, reverse=not is_ascending)
            elif key == "server_name":
                result.sort(key=lambda ep: ep.server_name, reverse=not is_ascending)
            elif key == "created_at":
                result.sort(key=lambda ep: ep.creation_timestamp or 0, reverse=not is_ascending)
            elif key == "last_updated_at":
                result.sort(key=lambda ep: ep.last_updated_timestamp or 0, reverse=not is_ascending)
        return result

    def update_mcp_access_endpoint(
        self,
        server_name: str,
        endpoint_id: str,
        server_version: str | None = _NOT_SET,
        server_alias: str | None = _NOT_SET,
        url: str | None = _NOT_SET,
        transport_type: MCPRemoteTransportType | None = _NOT_SET,
        last_updated_by: str | None = None,
    ) -> MCPAccessEndpoint:
        if server_version is not NOT_SET and server_alias is not NOT_SET:
            if server_version is not None and server_alias is not None:
                raise MlflowException(
                    "Cannot set both server_version and server_alias in a single update",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        client = self._mcp_client()
        res = client.get_mcpaccess_endpoint(
            GetMCPAccessEndpointParams(scope=_SCOPE, endpoint_id=endpoint_id)
        )
        if isinstance(res.value, GetMCPAccessEndpointMissing) or not isinstance(
            res.value, GetMCPAccessEndpointFound
        ):
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        ep = res.value.endpoint
        if ep.server_name != server_name:
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} does not belong to server '{server_name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        server_entity = self._fetch_mcp_server_entity(server_name)
        new_version = ep.server_version
        new_alias = ep.server_alias
        if server_version is not NOT_SET and server_version is not None:
            sv = self._fetch_mcp_server_version_entity(
                server_name, server_version, include_deleted=True
            )
            if sv.status == MCPStatus.DELETED.value:
                raise MlflowException(
                    f"Cannot update MCP access endpoint to deleted "
                    f"MCP server version '{server_name}' version '{server_version}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            new_version = server_version
            new_alias = None
        if server_alias is not NOT_SET and server_alias is not None:
            self._validate_alias_target_version(server_entity, server_alias)
            new_alias = server_alias
            new_version = None
        new_url = ep.url
        if url is not NOT_SET:
            if url is None:
                raise MlflowException(
                    "MCP access endpoint url cannot be None",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            _validate_mcp_access_endpoint_url(url)
            new_url = url
        new_transport = ep.transport_type
        if transport_type is not NOT_SET and transport_type is not None:
            new_transport = transport_type.value
        up_res = client.upsert_mcpaccess_endpoint_entity(
            UpsertMCPAccessEndpointEntityInput(
                request_id=str(uuid.uuid4()),
                endpoint_id=ep.endpoint_id,
                server_name=ep.server_name,
                url=new_url,
                transport_type=new_transport,
                server_version=new_version,
                server_alias=new_alias,
                created_by=ep.created_by,
                last_updated_by=last_updated_by,
                created_at_ms=int(ep.created_at_ms),
                last_updated_at_ms=_now_ms(),
            )
        )
        if not isinstance(up_res.outcome, UpsertMCPAccessEndpointEntityMCPAccessEndpointUpserted):
            raise MlflowException("Invalid UpsertMCPAccessEndpointEntity response from RiffDB")
        resolved = self._resolve_endpoint_version(
            up_res.outcome.endpoint, server_entity=server_entity
        )
        assert resolved is not None
        return self._endpoint_entity_to_mlflow(up_res.outcome.endpoint, resolved)

    def delete_mcp_access_endpoint(self, server_name: str, endpoint_id: str) -> None:
        client = self._mcp_client()
        res = client.get_mcpaccess_endpoint(
            GetMCPAccessEndpointParams(scope=_SCOPE, endpoint_id=endpoint_id)
        )
        if isinstance(res.value, GetMCPAccessEndpointMissing) or not isinstance(
            res.value, GetMCPAccessEndpointFound
        ):
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if res.value.endpoint.server_name != server_name:
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} does not belong to server '{server_name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        del_res = client.delete_mcpaccess_endpoint_entity(
            DeleteMCPAccessEndpointEntityInput(
                request_id=str(uuid.uuid4()),
                endpoint_id=endpoint_id,
            )
        )
        if isinstance(del_res.outcome, DeleteMCPAccessEndpointEntityMCPAccessEndpointMissing):
            raise MlflowException(
                f"MCPAccessEndpoint {endpoint_id} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        if not isinstance(del_res.outcome, DeleteMCPAccessEndpointEntityMCPAccessEndpointDeleted):
            raise MlflowException("Invalid DeleteMCPAccessEndpointEntity response from RiffDB")

    # --- Tag operations ---

    def set_mcp_server_tag(self, name: str, key: str, value: str) -> None:
        entity = self._fetch_mcp_server_entity(name)
        tags = {str(k): str(v) for k, v in _unpack_dict(entity.packed_tags).items()}
        tags[key] = value
        self._save_mcp_server_entity(
            MCPServerEntity(
                scope=_SCOPE,
                name=entity.name,
                display_name=entity.display_name,
                description=entity.description,
                packed_icons=entity.packed_icons,
                packed_tags=_pack_json(tags),
                packed_aliases=entity.packed_aliases,
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
                created_at_ms=entity.created_at_ms,
                last_updated_at_ms=entity.last_updated_at_ms,
            )
        )

    def delete_mcp_server_tag(self, name: str, key: str) -> None:
        entity = self._fetch_mcp_server_entity(name)
        tags = {str(k): str(v) for k, v in _unpack_dict(entity.packed_tags).items()}
        if key not in tags:
            raise MlflowException(
                f"Tag '{key}' not found on MCP server '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        tags.pop(key)
        self._save_mcp_server_entity(
            MCPServerEntity(
                scope=_SCOPE,
                name=entity.name,
                display_name=entity.display_name,
                description=entity.description,
                packed_icons=entity.packed_icons,
                packed_tags=_pack_json(tags),
                packed_aliases=entity.packed_aliases,
                created_by=entity.created_by,
                last_updated_by=entity.last_updated_by,
                created_at_ms=entity.created_at_ms,
                last_updated_at_ms=entity.last_updated_at_ms,
            )
        )

    def set_mcp_server_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        sv = self._fetch_mcp_server_version_entity(name, version, include_deleted=False)
        tags = {str(k): str(v) for k, v in _unpack_dict(sv.packed_tags).items()}
        tags[key] = value
        self._save_mcp_server_version_entity(
            MCPServerVersionEntity(
                scope=_SCOPE,
                name=sv.name,
                server_version=sv.server_version,
                version_major=sv.version_major,
                version_minor=sv.version_minor,
                version_patch=sv.version_patch,
                version_prerelease_sort_key=sv.version_prerelease_sort_key,
                server_json=sv.server_json,
                status=sv.status,
                packed_tools=sv.packed_tools,
                packed_tags=_pack_json(tags),
                packed_connect_options=sv.packed_connect_options,
                source_url=sv.source_url,
                created_by=sv.created_by,
                last_updated_by=sv.last_updated_by,
                created_at_ms=sv.created_at_ms,
                last_updated_at_ms=sv.last_updated_at_ms,
            )
        )

    def delete_mcp_server_version_tag(self, name: str, version: str, key: str) -> None:
        sv = self._fetch_mcp_server_version_entity(name, version, include_deleted=False)
        tags = {str(k): str(v) for k, v in _unpack_dict(sv.packed_tags).items()}
        if key not in tags:
            raise MlflowException(
                f"Tag '{key}' not found on MCP server version '{name}' version '{version}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        tags.pop(key)
        self._save_mcp_server_version_entity(
            MCPServerVersionEntity(
                scope=_SCOPE,
                name=sv.name,
                server_version=sv.server_version,
                version_major=sv.version_major,
                version_minor=sv.version_minor,
                version_patch=sv.version_patch,
                version_prerelease_sort_key=sv.version_prerelease_sort_key,
                server_json=sv.server_json,
                status=sv.status,
                packed_tools=sv.packed_tools,
                packed_tags=_pack_json(tags),
                packed_connect_options=sv.packed_connect_options,
                source_url=sv.source_url,
                created_by=sv.created_by,
                last_updated_by=sv.last_updated_by,
                created_at_ms=sv.created_at_ms,
                last_updated_at_ms=sv.last_updated_at_ms,
            )
        )

    # --- Alias operations ---

    def set_mcp_server_alias(self, name: str, alias: str, version: str) -> None:
        if alias == "latest":
            raise MlflowException(
                "The alias name 'latest' is reserved for automatic resolution",
                error_code=INVALID_PARAMETER_VALUE,
            )
        server_entity = self._fetch_mcp_server_entity(name)
        sv = self._fetch_mcp_server_version_entity(name, version, include_deleted=True)
        if sv.status == MCPStatus.DELETED.value:
            raise MlflowException(
                f"Cannot set alias '{alias}' to deleted MCP server version "
                f"'{name}' version '{version}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        aliases_map[alias] = version
        self._save_mcp_server_entity(
            MCPServerEntity(
                scope=_SCOPE,
                name=server_entity.name,
                display_name=server_entity.display_name,
                description=server_entity.description,
                packed_icons=server_entity.packed_icons,
                packed_tags=server_entity.packed_tags,
                packed_aliases=_pack_json(aliases_map),
                created_by=server_entity.created_by,
                last_updated_by=server_entity.last_updated_by,
                created_at_ms=server_entity.created_at_ms,
                last_updated_at_ms=server_entity.last_updated_at_ms,
            )
        )

    def delete_mcp_server_alias(self, name: str, alias: str) -> None:
        server_entity = self._fetch_mcp_server_entity(name)
        aliases_map = {
            str(k): str(v) for k, v in _unpack_dict(server_entity.packed_aliases).items()
        }
        if alias not in aliases_map:
            raise MlflowException(
                f"Alias '{alias}' not found on MCP server '{name}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        aliases_map.pop(alias)
        client = self._mcp_client()
        for ep in self._list_mcp_access_endpoint_entities(name):
            if ep.server_alias == alias:
                client.delete_mcpaccess_endpoint_entity(
                    DeleteMCPAccessEndpointEntityInput(
                        request_id=str(uuid.uuid4()),
                        endpoint_id=ep.endpoint_id,
                    )
                )
        self._save_mcp_server_entity(
            MCPServerEntity(
                scope=_SCOPE,
                name=server_entity.name,
                display_name=server_entity.display_name,
                description=server_entity.description,
                packed_icons=server_entity.packed_icons,
                packed_tags=server_entity.packed_tags,
                packed_aliases=_pack_json(aliases_map),
                created_by=server_entity.created_by,
                last_updated_by=server_entity.last_updated_by,
                created_at_ms=server_entity.created_at_ms,
                last_updated_at_ms=server_entity.last_updated_at_ms,
            )
        )
