from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import logging

logger = logging.getLogger(__name__)

class MCPClientManager:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self._mcp_tools: list = []        # LangChain-compatible tools
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        servers = {
            "math": "app/mcp/servers/math_server.py",
            # Add more MCP servers here as you build them
        }

        for name, path in servers.items():
            await self._connect(name, path)

        self._initialized = True
        logger.info(f"MCP initialized — {len(self._mcp_tools)} tools loaded")

    async def _connect(self, server_name: str, script_path: str):
        try:
            params = StdioServerParameters(command="python", args=[script_path])
            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # ✅ This converts MCP tools → LangChain tools automatically
            tools = await load_mcp_tools(session)
            self._mcp_tools.extend(tools)
            self.sessions[server_name] = session

            logger.info(f"Connected to '{server_name}' — tools: {[t.name for t in tools]}")

        except Exception as e:
            logger.error(f"Failed to connect to '{server_name}': {e}")
            raise

    def get_tools(self) -> list:
        """Return MCP tools as LangChain tools — merge with your existing LANGCHAIN_TOOLS."""
        return self._mcp_tools

    async def shutdown(self):
        await self.exit_stack.aclose()
        logger.info("MCP connections closed")


mcp_client = MCPClientManager()