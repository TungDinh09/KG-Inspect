
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, AsyncIterator, Iterator, List, Optional

from lightrag import LightRAG 
from lightrag.base import QueryParam
from lightrag.utils import always_get_an_event_loop, logger
from lightrag.prompt import PROMPTS
from kg_inspect.operate import (
    kg_query,
    naive_query,
)

class KGInspect(LightRAG):
    
    def query(
        self,
        query: str,
        images: Optional[List[str]] = None,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, images, param, system_prompt))  # type: ignore
    
    async def aquery(
        self,
        query: str,
        images: Optional[List[str]] = None,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query (backward compatibility wrapper).

        This function is now a wrapper around aquery_llm that maintains backward compatibility
        by returning only the LLM response content in the original format.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            system_prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str | AsyncIterator[str]: The LLM response content.
                - Non-streaming: Returns str
                - Streaming: Returns AsyncIterator[str]
        """
        # Call the new aquery_llm function to get complete results
        result = await self.aquery_vlm(query, images, param, system_prompt)

        # Extract and return only the LLM response for backward compatibility
        llm_response = result.get("llm_response", {})

        if llm_response.get("is_streaming"):
            return llm_response.get("response_iterator")
        else:
            return llm_response.get("content", "")
        
    
    def query_vlm(self,
        query: str,
        images: Optional[List[str]] = None,
        param: QueryParam = QueryParam(),
        system_prompt: Optional[str] = None,):
        """
        Synchronous complete query API: returns structured retrieval results with LLM generation.

        This function is the synchronous version of aquery_llm, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            system_prompt: Optional custom system prompt for LLM generation.
            images: Optional list of image paths for visual language model input.
        Returns:
            dict[str, Any]: Same complete response format as aquery_llm.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_vlm(query, images, param, system_prompt))
    
    async def aquery_vlm(self,
        query: str,
        images: Optional[List[str]] = None,
        param: QueryParam = QueryParam(),
        system_prompt: Optional[str] = None,
    )-> dict[str, Any]:
        """
        Asynchronous complete query API: returns structured retrieval results with LLM generation.

        This function performs a single query operation and returns both structured data and LLM response,
        based on the original aquery logic to avoid duplicate calls.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            dict[str, Any]: Complete response with structured data and LLM response.
        """
        logger.debug(f"[aquery_llm] Query param: {param}")

        global_config = asdict(self)

        try:
            query_result = None

            if param.mode in ["local", "global", "hybrid", "mix"]:
                query_result = await kg_query(
                    query.strip(),
                    self.chunk_entity_relation_graph,
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.text_chunks,
                    param,
                    global_config,
                    images=images,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                    chunks_vdb=self.chunks_vdb,
                )
            elif param.mode == "naive":
                query_result = await naive_query(
                    query.strip(),
                    self.chunks_vdb,
                    param,
                    global_config,
                    images=images,
                    hashing_kv=self.llm_response_cache,
                    system_prompt=system_prompt,
                )
            elif param.mode == "bypass":
                # Bypass mode: directly use LLM without knowledge retrieval
                use_llm_func = param.model_func or global_config["llm_model_func"]
                # Apply higher priority (8) to entity/relation summary tasks
                use_llm_func = partial(use_llm_func, _priority=8)

                param.stream = True if param.stream is None else param.stream
                response = await use_llm_func(
                    query.strip(),
                    images=images,
                    system_prompt=system_prompt,
                    history_messages=param.conversation_history,
                    enable_cot=True,
                    stream=param.stream,
                )
                if type(response) is str:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM non streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": response,
                            "response_iterator": None,
                            "is_streaming": False,
                        },
                    }
                else:
                    return {
                        "status": "success",
                        "message": "Bypass mode LLM streaming response",
                        "data": {},
                        "metadata": {},
                        "llm_response": {
                            "content": None,
                            "response_iterator": response,
                            "is_streaming": True,
                        },
                    }
            else:
                raise ValueError(f"Unknown mode {param.mode}")

            await self._query_done()

            # Check if query_result is None
            if query_result is None:
                return {
                    "status": "failure",
                    "message": "Query returned no results",
                    "data": {},
                    "metadata": {
                        "failure_reason": "no_results",
                        "mode": param.mode,
                    },
                    "llm_response": {
                        "content": PROMPTS["fail_response"],
                        "response_iterator": None,
                        "is_streaming": False,
                    },
                }

            # Extract structured data from query result
            raw_data = query_result.raw_data or {}
            raw_data["llm_response"] = {
                "content": query_result.content
                if not query_result.is_streaming
                else None,
                "response_iterator": query_result.response_iterator
                if query_result.is_streaming
                else None,
                "is_streaming": query_result.is_streaming,
            }

            return raw_data

        except Exception as e:
            logger.error(f"Query failed: {e}")
            # Return error response
            return {
                "status": "failure",
                "message": f"Query failed: {str(e)}",
                "data": {},
                "metadata": {},
                "llm_response": {
                    "content": None,
                    "response_iterator": None,
                    "is_streaming": False,
                },
            }