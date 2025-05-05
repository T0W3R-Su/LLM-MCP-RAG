import os
from mcp import Tool
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionToolParam
from openai.types import FunctionDefinition

from dataclasses import dataclass, field

from utils.pretty import log_title 

class ToolCallFunction:
    """工具调用函数类
    """
    name: str = ""
    arguments: str = ""

class ToolCall:
    """工具调用类
    基于 OpenAI API 的工具调用功能，封装了工具调用的 ID 和函数
    """
    id: str
    function: ToolCallFunction = ToolCallFunction()

class ChatOpenAIResponse():
    """ChatOpenAI 响应类
    """
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

@dataclass
class chatOpenAI:
    llm: AsyncOpenAI = field(init=False)

    model: str
    tools: list[Tool] = field(default_factory=list)

    system_prompt: str = ""
    context: str = ""
    messages: list[ChatCompletionMessage] = field(default_factory=list)

    def __post_init__(self):
        self.llm = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
        )
        if self.system_prompt:
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})

        if self.context:
            self.messages.append(
                {"role": "user", "content": self.context}
            )

    async def chat(self, prompt: str) -> str:
        log_title("CHAT")
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        
        # 接收流式响应
        streaming = await self.llm.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.getToolsDefinition(),
            stream=True,
        )

        content = ""
        toolCalls: list[ToolCall] = [] # 存储工具调用序列

        log_title("RESPONSE")
        async for chunk in streaming:
            delta = chunk.choices[0].delta
            # 处理 content
            if delta.content:
                content += delta.content or "" # 避免 None 值
                print(delta.content, end="", flush=True) # 流式传输输出

            # 处理 toolCalls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if (len(toolCalls) <= tool_call.index):
                        # 如果 toolCalls 列表长度小于 tool_call.index 收到新 tool_call
                        # 待流式传输补充
                        toolCalls.append(ToolCall())

                    current_tool_call = toolCalls[tool_call.index] # 当前调用的工具
                    if tool_call.id:
                        current_tool_call.id = tool_call.id
                    if tool_call.function:
                        current_tool_call.function.name = tool_call.function.name or ""
                        current_tool_call.function.arguments = (tool_call.function.arguments) or ""

        # 维护 messages 列表
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in toolCalls
                ],
            }
        )

        return ChatOpenAIResponse(
            content=content,
            tool_calls=toolCalls,
        )
    

    def getToolsDefinition(self):
        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ),
            )
            for tool in self.tools
        ]

async def example():
    llm = chatOpenAI(
        model = "gpt-4o-mini",
    )
    response = await llm.chat("Hello, how are you?")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(example())