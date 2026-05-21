#!/usr/bin/env python3
"""
pgEdge Postgres MCP Chatbot Client (HTTP + Gemini LLM + Raw Ollama Embeddings)

This script implements an asynchronous command-line chatbot agent that bridges 
cloud-based reasoning (Google Gemini) with local database capabilities and 
embeddings (via the pgEdge Model Context Protocol server and local Ollama).

Architecture Blueprint:
                   +-------------------------+
                   |       Chatbot.py        |
                   +------------+------------+
                                |
       (1) LLM Reason / Chat   |   (2) Tool Execution / Context
                 v              v
    +------------------------+  +---------------------------------+
    | Google Gemini Cloud API|  | Local pgEdge MCP Server (:8080) |
    +------------------------+  +----------------+----------------+
                                                 |
                                                 v
                                    +------------+------------+
                                    | Local Postgres Database |
                                    +-------------------------+

Dependencies:
    - google-genai (>= 0.28.1)
    - httpx (>= 0.28.1)

Usage:
    export GEMINI_API_KEY="AIzaSy..."
    export PGEDGE_MCP_SERVER_URL="http://localhost:8080/mcp/v1"
    python chatbot.py
"""

import asyncio
import json
import os
from typing import Optional, List, Dict, Any

import httpx
from google import genai
from google.genai import types


class PostgresChatbot:
    """
    An agentic chatbot class that orchestrates tool-calling workflows between
    Google Gemini and an active pgEdge PostgreSQL MCP server.
    """
    
    def __init__(self):
        """
        Initializes API clients, pulls configurations from environment variables,
        and provisions internal connection pools.
        """
        # Automatically load environment variables from the .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # Fallback if python-dotenv isn't installed; relies on shell-exported vars
            pass

        # 1. Look for the specific pgEdge environment variable mappings first
        api_key = (
            os.getenv("PGEDGE_GEMINI_API_KEY") or 
            os.getenv("GEMINI_API_KEY") or 
            os.getenv("PGEDGE_OPENAI_API_KEY") or
            os.getenv("OPENAI_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "No API key detected! Please ensure PGEDGE_GEMINI_API_KEY is set in your .env file."
            )
        
        # Initialize the Google Gemini client
        self.gemini_client = genai.Client(api_key=api_key)
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # 2. Configure local Ollama settings for vector embedding fallback pipelines
        # Using PGEDGE_OLLAMA_URL to match your .env file template
        self.ollama_base_url = os.getenv("PGEDGE_OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # 3. Establish the HTTP target destination for the pgEdge MCP layer
        self.mcp_server_url = os.getenv("PGEDGE_MCP_SERVER_URL", "http://localhost:8080/mcp/v1")
        if not self.mcp_server_url:
            raise ValueError("PGEDGE_MCP_SERVER_URL environment variable is required")

        # Initialize an Asynchronous HTTP Client to reuse TCP connection slots
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.request_id = 0

    def _get_next_id(self) -> int:
        """
        Increments and returns the next tracking identifier for JSON-RPC payloads.
        """
        self.request_id += 1
        return self.request_id

    async def _jsonrpc_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a standard structured JSON-RPC 2.0 transaction against the MCP server.
        """
        request_payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params or {}
        }

        try:
            response = await self.http_client.post(
                self.mcp_server_url,
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise Exception(f"JSON-RPC error payload received: {result['error']}")

            return result.get("result", {})
        except httpx.HTTPError as e:
            raise Exception(f"Failed to communicate with pgEdge server: {e}")

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        Queries the active MCP endpoint to request a registry manifest of structural 
        database capabilities exposed by the server.
        """
        try:
            result = await self._jsonrpc_request("tools/list")
            return result.get("tools", [])
        except Exception as e:
            print(f"Error querying tool structural layouts: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches an abstract operational directive back to the local backend 
        for evaluation against the connected PostgreSQL database state.
        """
        try:
            result = await self._jsonrpc_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            return result
        except Exception as e:
            return {
                "error": str(e),
                "is_error": True
            }

    def format_tools_for_gemini(self, tools: List[Dict[str, Any]]) -> str:
        """
        Aggregates raw tool manifests into clean context blocks readable inside
        the model's runtime execution prompt.
        """
        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- {tool['name']}: {tool.get('description', 'No description given.')}"
            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                if 'properties' in schema:
                    params = []
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', '')
                        params.append(f"  * {param_name} ({param_type}): {param_desc}")
                    if params:
                        tool_desc += "\n  Parameters Required:\n" + "\n".join(params)
            tool_descriptions.append(tool_desc)
        return "\n".join(tool_descriptions)

    def clean_json_fences(self, text: str) -> str:
        """
        Defensive Parser: Trims and sanitizes markdown code blocks (```json ... ```) 
        frequently returned by chat-centric models to guarantee clean JSON text feeds.
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Pop off the starting tag segment line (e.g., ```json or ```)
            cleaned = cleaned.split("\n", 1)[-1]
            # Strip off the ending markdown segment line
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        return cleaned.strip()

    async def process_query(self, user_query: str, chat_history: List[types.Content], tools: List[Dict[str, Any]]) -> str:
        """
        Manages the execution lifecycle loop of input requests, evaluating Gemini 
        reasoning loops and forwarding JSON-RPC parameters into target tool slots.
        """
        tools_context = self.format_tools_for_gemini(tools)

        # Enforce rigid behavior characteristics via system tracking controls
        system_instruction = f"""You are a strict PostgreSQL database agent.
        
AVAILABLE DATABASE TOOLS:
{tools_context}

CRITICAL INSTRUCTIONS:
1. Every time the user asks a database question, your ONLY allowed action is to select a tool and respond with a clean, raw JSON block matching this layout:
{{
    "tool": "tool_name",
    "arguments": {{
        "argument_name": "value"
    }}
}}
2. If you do not have sufficient database context or schema knowledge, immediately call 'get_schema_info' or related discovery tools.
3. Once you receive a "Tool result" block from the runtime environment, compile a final, user-friendly conversational summary using ONLY the facts returned."""

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0  # Set to absolute zero to prevent structural syntax or query logic hallucinations
        )

        # Update the long-term context logging trace
        chat_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
        )

        # Cap iteration state space to prevent unbounded infinite execution spirals
        for _ in range(10):
            # Run model generation safely in a background thread to prevent halting the async event thread loop
            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model=self.gemini_model,
                contents=chat_history,
                config=config
            )

            assistant_message = response.text
            
            # Extract raw string parameters inside code text boundaries
            cleaned_message = self.clean_json_fences(assistant_message)

            try:
                # Attempt to parse as an intentional action directive
                tool_call = json.loads(cleaned_message)

                if 'tool' in tool_call:
                    tool_name = tool_call['tool']
                    tool_args = tool_call.get('arguments', {})

                    print(f"  → Executing tool: {tool_name}")
                    tool_result = await self.call_tool(tool_name, tool_args)

                    # Mirror the tracking chains cleanly back to memory logs
                    chat_history.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=assistant_message)])
                    )
                    
                    # Feed execution facts right back down the inference stack
                    result_text = json.dumps(tool_result, indent=2)
                    chat_history.append(
                        types.Content(role="user", parts=[types.Part.from_text(text=f"Tool result:\n{result_text}")])
                    )
                    
                    # Tool executed successfully. Continue generation loop to feed context back to Gemini.
                    continue

            except json.JSONDecodeError:
                # Execution structure failed structural parsing, treat it as conversational text
                pass

            # Update final contextual responses
            chat_history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=assistant_message)])
            )
            return assistant_message

        return "Maximum tool execution loops exceeded. Try clarifying your input phrase."

    async def chat_loop(self):
        """
        Manages the interactive command-line interface thread execution environment.
        """
        print("\nPostgreSQL Chatbot (Google Gemini Production Edition)")
        print("=" * 60)

        tools = await self.list_available_tools()
        if not tools:
            print("\nWarning: No tools could be fetched from the MCP server. Verify port 8080 or check auth flags.")
            return

        chat_history: List[types.Content] = []

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nTerminating agent framework. Goodbye!")
                    break

                if not user_input:
                    continue

                print()
                response = await self.process_query(user_input, chat_history, tools)
                print(f"Assistant: {response}")

            except KeyboardInterrupt:
                print("\n\nTerminating agent framework. Goodbye!")
                break
            except Exception as e:
                print(f"\nError encountered inside evaluation block: {e}")

    async def cleanup(self):
        """Closes downstream socket connections securely."""
        await self.http_client.aclose()


async def main():
    """
    Main runtime bootstrap vector.
    """
    mcp_server_url = os.getenv("PGEDGE_MCP_SERVER_URL", "http://localhost:8080/mcp/v1")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    print(f"LLM Engine: Google Gemini ({gemini_model}) via Cloud API")
    print(f"Embedding Engine: Local Ollama Model Target ({os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')})")
    print(f"MCP Target Server: {mcp_server_url}")

    chatbot = PostgresChatbot()
    try:
        print("✓ Systems Ready. Framework Online.")
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass