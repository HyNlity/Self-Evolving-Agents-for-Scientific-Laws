"""测试 Agent 的 Context 管理功能

测试 agent.py 中所有与上下文管理相关的功能：
- reset_context()
- add_user_message()
- add_assistant_message()
- add_tool_message()
- get_current_dialog()
- get_conversation_history()
- context_manager.prepare_for_query()
- context_manager.should_truncate()
- context_manager.truncate() (不同策略)
- context_manager.estimate_tokens()
"""
# 添加项目根目录到 Python 路径，以便导入 evomaster 模块
import sys
import json
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from evomaster.agent.agent import Agent, AgentConfig
from evomaster.agent.context import ContextConfig, ContextManager, TruncationStrategy
from evomaster.agent.session.base import BaseSession, SessionConfig
from evomaster.agent.tools.builtin.bash import BashTool
from evomaster.agent.tools.builtin.finish import FinishTool
from evomaster.agent.tools.base import ToolRegistry
from evomaster.utils.types import (
    AssistantMessage,
    Dialog,
    SystemMessage,
    TaskInstance,
    ToolCall,
    ToolMessage,
    UserMessage,
    FunctionCall,
)


class MockLLM:
    """Mock LLM 用于测试"""
    
    def __init__(self):
        self.query_calls = []
    
    def query(self, dialog: Dialog) -> AssistantMessage:
        """模拟 LLM 查询"""
        self.query_calls.append(dialog)
        # 返回一个简单的助手消息
        return AssistantMessage(content="Mock response")


class MockSession(BaseSession):
    """Mock Session 用于测试"""
    
    def __init__(self, config=None):
        super().__init__(config or SessionConfig())
        self._is_open = True
    
    def open(self):
        self._is_open = True
    
    def close(self):
        self._is_open = False
    
    def exec_bash(self, command: str, timeout=None, is_input=False):
        return {"stdout": "", "stderr": "", "exit_code": 0}
    
    def upload(self, local_path: str, remote_path: str):
        pass
    
    def download(self, remote_path: str, timeout=None):
        return b""


class TestAgentContextManagement(unittest.TestCase):
    """测试 Agent 的 Context 管理功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.llm = MockLLM()
        self.session = MockSession()
        self.tools = ToolRegistry()
        self.task = TaskInstance(
            task_id="test_task_001",
            task_type="test",
            description="测试任务",
            input_data={}
        )
        
        # 创建临时提示词文件（在测试期间保持存在）
        self.tmpdir = tempfile.mkdtemp()
        tmpdir_path = Path(self.tmpdir)
        self.system_prompt_file = tmpdir_path / "system_prompt.txt"
        self.user_prompt_file = tmpdir_path / "user_prompt.txt"
        
        self.system_prompt_file.write_text("You are a test assistant.")
        self.user_prompt_file.write_text("Complete the test task.")
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        if hasattr(self, 'tmpdir'):
            shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def create_agent(self, context_config=None, enable_tools=False):
        """创建 Agent 实例"""
        agent_config = AgentConfig(context_config=context_config or ContextConfig())
        
        return Agent(
            llm=self.llm,
            session=self.session,
            tools=self.tools,
            system_prompt_file=str(self.system_prompt_file),
            user_prompt_file=str(self.user_prompt_file),
            config=agent_config,
            enable_tools=enable_tools,
        )
    
    def test_initial_context(self):
        """测试初始上下文状态"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # 检查初始对话
        dialog = agent.get_current_dialog()
        self.assertIsNotNone(dialog)
        self.assertEqual(len(dialog.messages), 2)  # System + User
        self.assertIsInstance(dialog.messages[0], SystemMessage)
        self.assertIsInstance(dialog.messages[1], UserMessage)
        
        # 检查初始提示词已保存
        self.assertIsNotNone(agent._initial_system_prompt)
        self.assertIsNotNone(agent._initial_user_prompt)
    
    def test_add_user_message(self):
        """测试添加用户消息"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.add_user_message("这是用户消息1")
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], UserMessage)
        self.assertEqual(history[-1].content, "这是用户消息1")
        
        # 再添加一条
        agent.add_user_message("这是用户消息2")
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 2)
        self.assertEqual(history[-1].content, "这是用户消息2")
    
    def test_add_assistant_message(self):
        """测试添加助手消息"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.add_assistant_message("这是助手回复")
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], AssistantMessage)
        self.assertEqual(history[-1].content, "这是助手回复")
        
        # 测试带工具调用的助手消息
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test_tool", arguments='{"arg": "value"}')
        )
        agent.add_assistant_message("调用工具", tool_calls=[tool_call])
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 2)
        self.assertIsNotNone(history[-1].tool_calls)
        self.assertEqual(len(history[-1].tool_calls), 1)
    
    def test_add_tool_message(self):
        """测试添加工具消息"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.add_tool_message(
            content="工具执行结果",
            tool_call_id="call_123",
            name="test_tool",
            meta={"status": "success"}
        )
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], ToolMessage)
        self.assertEqual(history[-1].content, "工具执行结果")
        self.assertEqual(history[-1].tool_call_id, "call_123")
        self.assertEqual(history[-1].name, "test_tool")
        self.assertEqual(history[-1].meta["status"], "success")
    
    def test_get_current_dialog(self):
        """测试获取当前对话"""
        agent = self.create_agent()
        
        # 初始化前应该返回 None
        self.assertIsNone(agent.get_current_dialog())
        
        # 初始化后应该返回 Dialog
        agent._initialize(self.task)
        dialog = agent.get_current_dialog()
        self.assertIsNotNone(dialog)
        self.assertIsInstance(dialog, Dialog)
        
        # 添加消息后，dialog 应该更新（get_current_dialog 返回的是同一个对象引用）
        initial_count = len(dialog.messages)
        agent.add_user_message("新消息")
        # 由于是同一个对象引用，直接检查当前 dialog
        self.assertEqual(len(agent.get_current_dialog().messages), initial_count + 1)
    
    def test_get_conversation_history(self):
        """测试获取对话历史"""
        agent = self.create_agent()
        
        # 初始化前应该返回空列表
        history = agent.get_conversation_history()
        self.assertEqual(history, [])
        
        # 初始化后应该有消息
        agent._initialize(self.task)
        history = agent.get_conversation_history()
        self.assertGreater(len(history), 0)
        
        # 添加消息后，历史应该更新
        agent.add_user_message("消息1")
        agent.add_assistant_message("回复1")
        agent.add_tool_message("结果1", "call_1", "tool1")
        
        history = agent.get_conversation_history()
        self.assertGreaterEqual(len(history), 4)  # 初始2条 + 3条新消息
    
    def test_reset_context(self):
        """测试重置上下文"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # 添加一些消息
        agent.add_user_message("消息1")
        agent.add_assistant_message("回复1")
        agent.add_user_message("消息2")
        
        initial_count = len(agent.get_conversation_history())
        self.assertGreater(initial_count, 2)  # 应该多于初始的2条
        
        # 重置上下文
        agent.reset_context()
        
        # 检查是否重置到初始状态
        history = agent.get_conversation_history()
        self.assertEqual(len(history), 2)  # 只有 System + User
        self.assertIsInstance(history[0], SystemMessage)
        self.assertIsInstance(history[1], UserMessage)
        
        # 检查步骤计数是否重置
        self.assertEqual(agent._step_count, 0)
    
    def test_reset_context_without_initialization(self):
        """测试未初始化时重置上下文应该报错"""
        agent = self.create_agent()
        
        with self.assertRaises(ValueError):
            agent.reset_context()
    
    def test_context_manager_estimate_tokens(self):
        """测试 token 估算"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        dialog = agent.get_current_dialog()
        tokens = agent.context_manager.estimate_tokens(dialog)
        
        # 应该返回一个非负整数
        self.assertIsInstance(tokens, int)
        self.assertGreaterEqual(tokens, 0)
        
        # 添加更多内容后，token 数应该增加
        agent.add_user_message("x" * 1000)  # 添加1000个字符
        new_dialog = agent.get_current_dialog()
        new_tokens = agent.context_manager.estimate_tokens(new_dialog)
        self.assertGreater(new_tokens, tokens)
    
    def test_context_manager_should_truncate(self):
        """测试是否需要截断的判断"""
        # 创建一个小 token 限制的配置
        context_config = ContextConfig(max_tokens=100)
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        dialog = agent.get_current_dialog()
        should_truncate = agent.context_manager.should_truncate(dialog)
        
        # 初始对话应该不需要截断
        self.assertFalse(should_truncate)
        
        # 添加大量内容
        for i in range(10):
            agent.add_user_message("x" * 1000)  # 每次添加1000字符
            agent.add_assistant_message("y" * 1000)
        
        new_dialog = agent.get_current_dialog()
        should_truncate = agent.context_manager.should_truncate(new_dialog)
        # 现在应该需要截断了
        self.assertTrue(should_truncate)
    
    def test_context_manager_prepare_for_query_no_truncation(self):
        """测试准备查询（不需要截断的情况）"""
        context_config = ContextConfig(max_tokens=1000000)  # 很大的限制
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        dialog = agent.get_current_dialog()
        prepared = agent.context_manager.prepare_for_query(dialog)
        
        # 不需要截断时，应该返回原始 dialog
        self.assertEqual(len(prepared.messages), len(dialog.messages))
        self.assertEqual(prepared.messages, dialog.messages)
    
    def test_context_manager_prepare_for_query_with_truncation(self):
        """测试准备查询（需要截断的情况）"""
        context_config = ContextConfig(
            max_tokens=50,  # 很小的限制，确保会触发截断
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加大量消息（每条消息约20字符，加上开销，10条消息约500字符，约125 tokens）
        for i in range(15):
            agent.add_user_message(f"用户消息 {i} " + "x" * 50)  # 增加消息长度
            agent.add_assistant_message(f"助手回复 {i} " + "y" * 50)
        
        dialog = agent.get_current_dialog()
        # 确保确实需要截断
        self.assertTrue(agent.context_manager.should_truncate(dialog))
        
        prepared = agent.context_manager.prepare_for_query(dialog)
        
        # 应该被截断了
        self.assertLess(len(prepared.messages), len(dialog.messages))
        # 应该保留系统消息
        self.assertIsInstance(prepared.messages[0], SystemMessage)
    
    def test_context_manager_truncate_none_strategy(self):
        """测试截断策略：NONE"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.NONE
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加消息
        for i in range(5):
            agent.add_user_message(f"消息 {i}")
        
        dialog = agent.get_current_dialog()
        truncated = agent.context_manager.truncate(dialog)
        
        # NONE 策略不应该截断
        self.assertEqual(len(truncated.messages), len(dialog.messages))
    
    def test_context_manager_truncate_latest_half_strategy(self):
        """测试截断策略：LATEST_HALF"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加多条助手消息（从助手消息开始截断）
        for i in range(8):
            agent.add_assistant_message(f"助手消息 {i}")
            agent.add_user_message(f"用户消息 {i}")
        
        dialog = agent.get_current_dialog()
        original_count = len(dialog.messages)
        
        truncated = agent.context_manager.truncate(dialog)
        
        # 应该被截断了
        self.assertLess(len(truncated.messages), original_count)
        # 应该保留系统消息
        self.assertIsInstance(truncated.messages[0], SystemMessage)
        # 应该保留用户初始消息
        self.assertIsInstance(truncated.messages[1], UserMessage)
        # 应该保留最新的一半
        self.assertGreater(len(truncated.messages), original_count // 2)
    
    def test_context_manager_truncate_sliding_window_strategy(self):
        """测试截断策略：SLIDING_WINDOW"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SLIDING_WINDOW,
            preserve_recent_turns=3
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加多条消息
        for i in range(10):
            agent.add_user_message(f"用户消息 {i}")
            agent.add_assistant_message(f"助手消息 {i}")
        
        dialog = agent.get_current_dialog()
        original_count = len(dialog.messages)
        
        truncated = agent.context_manager.truncate(dialog)
        
        # 应该被截断了
        self.assertLess(len(truncated.messages), original_count)
        # 应该保留系统消息
        self.assertIsInstance(truncated.messages[0], SystemMessage)
        # 应该保留最近的几轮对话（preserve_recent_turns=3，每轮约2-3条消息）
        # 所以应该保留约 1 (system) + 3*3 = 10 条消息左右
        self.assertLessEqual(len(truncated.messages), 15)  # 允许一些误差
    
    def test_context_manager_truncate_summary_strategy(self):
        """测试截断策略：SUMMARY（回退到 latest_half）"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SUMMARY
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加消息
        for i in range(8):
            agent.add_assistant_message(f"助手消息 {i}")
            agent.add_user_message(f"用户消息 {i}")
        
        dialog = agent.get_current_dialog()
        truncated = agent.context_manager.truncate(dialog)
        
        # SUMMARY 策略目前回退到 latest_half
        # 应该被截断了
        self.assertLess(len(truncated.messages), len(dialog.messages))
        # 检查 meta 信息
        self.assertIn("truncated", truncated.meta)
    
    def test_context_manager_with_token_counter(self):
        """测试使用自定义 token 计数器"""
        from evomaster.agent.context import SimpleTokenCounter
        
        token_counter = SimpleTokenCounter(chars_per_token=2.0)  # 每2个字符1个token
        context_config = ContextConfig(max_tokens=100)
        agent = self.create_agent(context_config)
        
        # 设置 token 计数器
        agent.context_manager.set_token_counter(token_counter)
        
        agent._initialize(self.task)
        dialog = agent.get_current_dialog()
        
        # 使用自定义计数器估算
        tokens = agent.context_manager.estimate_tokens(dialog)
        self.assertIsInstance(tokens, int)
        self.assertGreaterEqual(tokens, 0)
    
    def test_set_next_user_request(self):
        """测试 set_next_user_request 方法"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        initial_count = len(agent.get_conversation_history())
        agent.set_next_user_request("下一个用户请求")
        
        history = agent.get_conversation_history()
        self.assertEqual(len(history), initial_count + 1)
        self.assertIsInstance(history[-1], UserMessage)
        self.assertEqual(history[-1].content, "下一个用户请求")
    
    def test_context_preservation_after_reset(self):
        """测试重置后初始提示词是否保留"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # 保存初始提示词
        initial_system = agent._initial_system_prompt
        initial_user = agent._initial_user_prompt
        
        # 添加消息
        agent.add_user_message("消息1")
        agent.add_assistant_message("回复1")
        
        # 重置
        agent.reset_context()
        
        # 检查初始提示词仍然存在
        self.assertEqual(agent._initial_system_prompt, initial_system)
        self.assertEqual(agent._initial_user_prompt, initial_user)
        
        # 检查重置后的对话使用初始提示词
        history = agent.get_conversation_history()
        self.assertEqual(history[0].content, initial_system)
        self.assertEqual(history[1].content, initial_user)
    
    def test_multiple_resets(self):
        """测试多次重置"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # 第一次重置
        agent.add_user_message("消息1")
        agent.reset_context()
        self.assertEqual(len(agent.get_conversation_history()), 2)
        
        # 第二次重置
        agent.add_user_message("消息2")
        agent.add_assistant_message("回复2")
        agent.reset_context()
        self.assertEqual(len(agent.get_conversation_history()), 2)
        
        # 第三次重置
        agent.add_tool_message("结果", "call_1", "tool1")
        agent.reset_context()
        self.assertEqual(len(agent.get_conversation_history()), 2)
    
    def test_empty_message_content(self):
        """测试空消息内容"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # 添加空内容的消息
        agent.add_user_message("")
        agent.add_assistant_message(None)  # None 应该被允许
        
        history = agent.get_conversation_history()
        self.assertGreaterEqual(len(history), 4)  # 初始2条 + 2条新消息
        # 最后一条应该是助手消息
        self.assertIsInstance(history[-1], AssistantMessage)
    
    def test_very_large_message(self):
        """测试非常大的消息"""
        agent = self.create_agent()
        agent._initialize(self.task)
        
        # 创建一个非常大的消息（10000字符）
        large_content = "x" * 10000
        agent.add_user_message(large_content)
        
        history = agent.get_conversation_history()
        self.assertEqual(history[-1].content, large_content)
        
        # Token 估算应该能处理大消息
        dialog = agent.get_current_dialog()
        tokens = agent.context_manager.estimate_tokens(dialog)
        self.assertGreater(tokens, 1000)  # 应该估算出大量 tokens
    
    def test_truncate_with_no_assistant_messages(self):
        """测试没有助手消息时的截断"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 只添加用户消息，没有助手消息
        for i in range(5):
            agent.add_user_message(f"用户消息 {i}")
        
        dialog = agent.get_current_dialog()
        original_count = len(dialog.messages)
        
        # 截断应该能处理这种情况
        truncated = agent.context_manager.truncate(dialog)
        # 如果没有助手消息，latest_half 策略可能无法截断，返回原对话
        self.assertLessEqual(len(truncated.messages), original_count)
    
    def test_truncate_with_only_system_message(self):
        """测试只有系统消息时的截断"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SLIDING_WINDOW
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 创建一个只有系统消息的对话
        dialog = Dialog(messages=[SystemMessage(content="系统提示词")])
        
        # 截断应该保留系统消息
        truncated = agent.context_manager.truncate(dialog)
        self.assertEqual(len(truncated.messages), 1)
        self.assertIsInstance(truncated.messages[0], SystemMessage)
    
    def test_add_message_error_handling(self):
        """测试添加消息时的错误处理"""
        agent = self.create_agent()
        
        # 未初始化时添加消息应该报错
        with self.assertRaises(ValueError):
            agent.add_user_message("消息")
        
        with self.assertRaises(ValueError):
            agent.add_assistant_message("回复")
        
        with self.assertRaises(ValueError):
            agent.add_tool_message("结果", "call_1", "tool1")
        
        # 初始化后应该正常工作
        agent._initialize(self.task)
        agent.add_user_message("消息")
        self.assertEqual(len(agent.get_conversation_history()), 3)  # 初始2条 + 1条新消息
    
    def test_context_manager_preserve_system_messages(self):
        """测试保留系统消息的配置"""
        context_config = ContextConfig(
            max_tokens=100,
            truncation_strategy=TruncationStrategy.SLIDING_WINDOW,
            preserve_system_messages=True
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加多条消息
        for i in range(10):
            agent.add_user_message(f"用户消息 {i}")
            agent.add_assistant_message(f"助手消息 {i}")
        
        dialog = agent.get_current_dialog()
        truncated = agent.context_manager.truncate(dialog)
        
        # 应该保留系统消息
        self.assertIsInstance(truncated.messages[0], SystemMessage)
        # 系统消息内容应该保持不变
        self.assertEqual(truncated.messages[0].content, dialog.messages[0].content)
    
    def test_dialog_meta_preserved_after_truncation(self):
        """测试截断后保留原始 meta 信息"""
        # 创建需要截断的配置
        context_config = ContextConfig(
            max_tokens=50,  # 很小的限制，确保会触发截断
            truncation_strategy=TruncationStrategy.LATEST_HALF
        )
        agent = self.create_agent(context_config)
        agent._initialize(self.task)
        
        # 添加自定义 meta
        agent.current_dialog.meta["custom_key"] = "custom_value"
        
        # 添加大量消息（确保会触发截断）
        for i in range(15):
            agent.add_user_message(f"消息 {i} " + "x" * 50)
            agent.add_assistant_message(f"回复 {i} " + "y" * 50)
        
        dialog = agent.get_current_dialog()
        # 确保确实需要截断
        self.assertTrue(agent.context_manager.should_truncate(dialog))
        
        prepared = agent.context_manager.prepare_for_query(dialog)
        
        # 应该被截断了，所以应该有 truncated 标记
        self.assertIn("truncated", prepared.meta)
        # 应该保留原始的 meta
        self.assertEqual(prepared.meta.get("custom_key"), "custom_value")

    def test_recover_finish_tool_call_from_text_json(self):
        """测试从文本 JSON 恢复 finish 工具调用。"""
        self.tools.register(FinishTool())
        agent = self.create_agent(enable_tools=True)

        assistant_message = AssistantMessage(
            content='{"message":"done","task_completed":"true"}',
            tool_calls=None,
        )
        agent._recover_tool_calls_from_text(assistant_message)

        self.assertIsNotNone(assistant_message.tool_calls)
        assert assistant_message.tool_calls is not None
        self.assertEqual(len(assistant_message.tool_calls), 1)
        self.assertEqual(assistant_message.tool_calls[0].function.name, "finish")
        args = json.loads(assistant_message.tool_calls[0].function.arguments)
        self.assertEqual(args["message"], "done")
        self.assertEqual(args["task_completed"], "true")

    def test_recover_execute_bash_from_text_json(self):
        """测试从文本 JSON 恢复 execute_bash 工具调用。"""
        self.tools.register(BashTool())
        agent = self.create_agent(enable_tools=True)

        assistant_message = AssistantMessage(
            content='{"command":"pwd"}',
            tool_calls=None,
        )
        agent._recover_tool_calls_from_text(assistant_message)

        self.assertIsNotNone(assistant_message.tool_calls)
        assert assistant_message.tool_calls is not None
        self.assertEqual(len(assistant_message.tool_calls), 1)
        self.assertEqual(assistant_message.tool_calls[0].function.name, "execute_bash")
        args = json.loads(assistant_message.tool_calls[0].function.arguments)
        self.assertEqual(args["command"], "pwd")


if __name__ == "__main__":
    unittest.main()
