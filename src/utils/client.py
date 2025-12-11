from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from typing import Any
import json

llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0)

class ChatDeepSeekWithReasoning(ChatDeepSeek):
    """扩展 ChatDeepSeek 以支持 deepseek-reasoner 模型的 reasoning_content 字段"""
    
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        # 第一步：在转换之前，从原始消息中提取 reasoning_content
        # 只收集 AIMessage 的 reasoning_content，按顺序存储
        reasoning_content_list = []
        
        # 处理不同格式的 input_
        messages = None
        if isinstance(input_, list):
            # input_ 直接是消息列表
            messages = input_
        elif isinstance(input_, dict) and "messages" in input_:
            # input_ 是字典，包含 messages 键
            messages = input_["messages"]
        
        # 从消息中提取 reasoning_content
        if messages:
            for msg in messages:
                if isinstance(msg, AIMessage):
                    # 从 additional_kwargs 中提取 reasoning_content
                    reasoning = None
                    if msg.additional_kwargs:
                        reasoning = msg.additional_kwargs.get("reasoning_content")
                    reasoning_content_list.append(reasoning)  # 可能是 None
        
        # 第二步：调用父类方法进行消息转换
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        # 第三步：处理转换后的消息，确保 reasoning_content 正确设置
        assistant_idx = 0  # 跟踪原始消息列表中 assistant 消息的索引
        
        for message in payload["messages"]:
            # 处理 tool 消息：将列表格式的 content 转为 JSON 字符串
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = json.dumps(message["content"])
            
            # 处理 assistant 消息：将列表格式的 content 转为字符串
            elif message["role"] == "assistant" and isinstance(
                message["content"], list
            ):
                # DeepSeek API expects assistant content to be a string, not a list.
                # Extract text blocks and join them, or use empty string if none exist.
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""
            
            # 处理 reasoning_content：当 assistant 消息有 tool_calls 时，必须包含 reasoning_content
            if message["role"] == "assistant":
                tool_calls = message.get("tool_calls")
                
                # 如果有 tool_calls（且不为空），必须包含 reasoning_content
                if tool_calls and len(tool_calls) > 0:
                    # 检查是否已经有 reasoning_content
                    if "reasoning_content" not in message:
                        # 尝试从列表中获取对应的 reasoning_content（如果有值的话）
                        # assistant_idx 对应 reasoning_content_list 中的索引
                        reasoning = None
                        if assistant_idx < len(reasoning_content_list):
                            reasoning = reasoning_content_list[assistant_idx]
                        
                        if reasoning is not None:
                            # 如果有值，使用它（这是从之前响应中获取的 reasoning_content）
                            message["reasoning_content"] = reasoning
                        else:
                            # 如果没有值，添加空字符串（满足 API 要求）
                            message["reasoning_content"] = ""
                    # 如果已经有 reasoning_content，保持原样（说明已经被正确包含）
                else:
                    # 如果没有 tool_calls，说明是最终答案或历史消息
                    # 根据官方文档建议，清除 reasoning_content 以节省带宽
                    # API 会忽略它，但清除可以节省网络传输
                    if "reasoning_content" in message:
                        message.pop("reasoning_content", None)
                
                assistant_idx += 1  # 移动到下一个 assistant 消息（只对 assistant 消息计数）
        
        # 第四步：移除 tool_choice 参数（deepseek-reasoner 不支持此参数）
        if "tool_choice" in payload:
            payload.pop("tool_choice", None)
        
        return payload


reason_llm = ChatDeepSeekWithReasoning(model="deepseek-reasoner", temperature=1.0)