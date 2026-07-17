import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_tool(tool_name: str, arguments: dict) -> dict:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server/server.py"],
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.model_dump()


def call_tool_sync(tool_name: str, arguments: dict) -> dict:
    return asyncio.run(call_tool(tool_name, arguments))