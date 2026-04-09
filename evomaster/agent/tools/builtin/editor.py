"""EvoMaster Editor 工具

提供文件查看、创建、编辑的能力。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from ..base import BaseTool, BaseToolParams, ToolError, ToolParameterError

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


# 截断提示
TEXT_FILE_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. '
    'You should retry this tool after you have searched inside the file with `grep -n` in order to find '
    'the line numbers of what you are looking for.</NOTE>'
)
DIRECTORY_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of this directory has been shown to you. '
    'You should use `ls -la` instead to view large directories incrementally.</NOTE>'
)

# 每次编辑显示的上下文行数
SNIPPET_LINES = 4
MAX_OUTPUT_SIZE = 16000
VIEW_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\t")
OVERWRITABLE_SCAFFOLD_MARKER = "EVO_SCAFFOLD_OVERWRITABLE"


def maybe_truncate(content: str, max_size: int = MAX_OUTPUT_SIZE, notice: str = TEXT_FILE_TRUNCATED_NOTICE) -> str:
    """中间截断内容"""
    if len(content) <= max_size:
        return content
    half = max_size // 2
    return content[:half] + "\n" + notice + "\n" + content[-half:]


class EditorToolParams(BaseToolParams):
    """Custom editing tool for viewing, creating and editing files in plain-text format.
    
    * State is persistent across command calls and discussions with the user
    * If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
    * The `undo_edit` command will revert the last edit made to the file at `path`

    Before using this tool:
    1. Use the view tool to understand the file's contents and context
    2. Verify the directory path is correct (only applicable when creating new files)

    When making edits:
       - Ensure the edit results in idiomatic, correct code
       - Do not leave the code in a broken state
       - Always use absolute file paths (starting with /)

    CRITICAL REQUIREMENTS FOR USING THIS TOOL:

    1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file. The tool will fail if `old_str` doesn't match exactly.

    2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
       - Include sufficient context before and after the change point (3-5 lines recommended)
       - If not unique, the replacement will not be performed

    3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.
    """
    
    name: ClassVar[str] = "str_replace_editor"

    command: Literal["view", "create", "str_replace", "insert", "undo_edit"] = Field(
        description="The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`."
    )
    path: str = Field(
        description="Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.",
    )
    file_text: str = Field(
        default="",
        description="Required parameter of `create` command, with the content of the file to be created.",
    )
    old_str: str = Field(
        default="",
        description="Required parameter of `str_replace` command containing the string in `path` to replace.",
    )
    new_str: str = Field(
        default="",
        description="Optional parameter of `str_replace` command containing the new string. Required parameter of `insert` command containing the string to insert.",
    )
    insert_line: int = Field(
        default=-1,
        description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
    )
    view_range: list[int] = Field(
        default_factory=list,
        description="Optional parameter of `view` command when `path` points to a file. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start.",
    )


class EditorTool(BaseTool):
    """文件编辑工具"""
    
    name: ClassVar[str] = "str_replace_editor"
    params_class: ClassVar[type[BaseToolParams]] = EditorToolParams

    def __init__(self):
        super().__init__()
        # 文件编辑历史 {path: [(content, encoding), ...]}
        self._file_history: dict[str, list[tuple[str, str]]] = {}

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行编辑操作"""
        try:
            params = self.parse_params(self._normalize_payload(args_json))
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, EditorToolParams)
        
        try:
            normalized_path = self._normalize_workspace_alias(session, params.path)
            # 验证路径
            path_type = self._validate_path(session, params.command, normalized_path)
            
            if params.command == "view":
                return self._view(session, normalized_path, params.view_range, path_type)
            elif params.command == "create":
                return self._create(session, normalized_path, params.file_text)
            elif params.command == "str_replace":
                return self._str_replace(session, normalized_path, params.old_str, params.new_str)
            elif params.command == "insert":
                return self._insert(session, normalized_path, params.insert_line, params.new_str)
            elif params.command == "undo_edit":
                return self._undo_edit(session, normalized_path)
            else:
                return f"Unknown command: {params.command}", {}
        except ToolError as e:
            return f"ERROR:\n{str(e)}", {"error": str(e)}

    def _normalize_payload(self, args_json: str) -> str:
        """Recover common gateway/model wrapper forms for str_replace_editor."""
        try:
            payload = json.loads(args_json)
        except Exception:
            return args_json
        if not isinstance(payload, dict):
            return args_json

        command = payload.get("command")
        if isinstance(command, str):
            stripped = command.strip()
            for prefix in ("str_replace_editor", "editor"):
                if stripped == prefix:
                    break
                if stripped.startswith(prefix + " "):
                    payload["command"] = stripped[len(prefix) + 1 :].strip()
                    break
                if stripped.startswith(prefix + "."):
                    payload["command"] = stripped[len(prefix) + 1 :].strip()
                    break

        return json.dumps(payload, ensure_ascii=False)

    def _resolve_workspace_root(self, session: BaseSession) -> str | None:
        """解析当前会话的真实工作空间根目录。"""
        get_workspace_path = getattr(session, "get_workspace_path", None)
        if callable(get_workspace_path):
            workspace_override = get_workspace_path()
            if workspace_override:
                return str(Path(workspace_override))

        config = getattr(session, "config", None)
        workspace_path = getattr(config, "workspace_path", None)
        if workspace_path:
            return str(Path(workspace_path))
        return None

    def _normalize_workspace_alias(self, session: BaseSession, path: str) -> str:
        """把通用 `/workspace/...` 别名或 workspace 内相对路径映射到真实工作空间。"""
        workspace_root = self._resolve_workspace_root(session)
        if not Path(path).is_absolute():
            if not workspace_root:
                return path
            workspace_path = Path(workspace_root).resolve()
            candidate = (workspace_path / path).resolve()
            if candidate == workspace_path or workspace_path in candidate.parents:
                return str(candidate)
            return path

        if not workspace_root or workspace_root == "/workspace":
            return path

        repaired_absolute = self._repair_workspace_absolute_path(workspace_root, path)
        if repaired_absolute != path:
            return repaired_absolute

        if path == "/workspace":
            return workspace_root
        if path.startswith("/workspace/"):
            relative_part = path.removeprefix("/workspace/")
            return str(Path(workspace_root) / relative_part)
        return path

    def _repair_workspace_absolute_path(self, workspace_root: str, path: str) -> str:
        """恢复被错误拼接过前缀的绝对路径。

        常见情况是模型把真实绝对路径又相对当前目录拼接了一次，导致类似
        `/home/.../SRAgent/SRAgent/repo/.../task_0/...` 这样的路径。只要能在目标
        路径里重新找到当前 workspace 的后缀，就优先恢复成真实 workspace 前缀。
        """
        workspace_path = Path(workspace_root).resolve()
        target_path = Path(path)
        if not target_path.is_absolute():
            return path
        if target_path == workspace_path or workspace_path in target_path.parents:
            return path

        incoming_parts = target_path.parts
        workspace_parts = workspace_path.parts
        for start in range(len(workspace_parts)):
            suffix = workspace_parts[start:]
            suffix_len = len(suffix)
            if not suffix:
                continue
            for index in range(len(incoming_parts) - suffix_len + 1):
                if incoming_parts[index : index + suffix_len] != suffix:
                    continue
                recovered = Path(*workspace_parts[:start], *incoming_parts[index:])
                if recovered == workspace_path or workspace_path in recovered.parents:
                    return str(recovered)
        return path

    def _validate_path(
        self,
        session: BaseSession,
        command: str,
        path: str,
    ) -> Literal["file", "dir", "not_exist"]:
        """验证路径"""
        # 检查是否是绝对路径
        if not Path(path).is_absolute():
            raise ToolParameterError("path", path, "The path should be an absolute path, starting with `/`.")
        
        # 检查路径类型（优先检查目录，因为目录检查更可靠）
        if session.is_directory(path):
            path_type = "dir"
        elif session.is_file(path):
            path_type = "file"
        elif session.path_exists(path):
            # 如果路径存在但既不是文件也不是目录，再次检查
            # 可能是符号链接或其他特殊类型，尝试判断实际类型
            if session.is_directory(path):
                path_type = "dir"
            elif session.is_file(path):
                path_type = "file"
            else:
                # 未知类型，默认当作文件处理（但会在使用时再次检查）
                path_type = "file"
        else:
            path_type = "not_exist"
        
        # 验证命令与路径类型的兼容性
        if path_type == "not_exist" and command != "create":
            raise ToolParameterError("path", path, f"The path {path} does not exist.")
        
        # 对于 create 命令，需要更严格的检查
        if command == "create":
            overwritable_scaffold = self._is_overwritable_scaffold(session, path)
            # 再次确认路径不存在（防止误判）
            if session.is_file(path):
                if not overwritable_scaffold:
                    raise ToolParameterError("path", path, f"File already exists at: {path}. Cannot overwrite files using command `create`.")
            if session.is_directory(path):
                raise ToolParameterError("path", path, f"The path {path} is a directory. Cannot create a file with the same name as a directory.")
            if session.path_exists(path) and not overwritable_scaffold:
                # 路径存在但不是文件也不是目录，可能是其他类型（如符号链接）
                raise ToolParameterError("path", path, f"Path already exists at: {path}. Cannot overwrite using command `create`.")
        
        if path_type == "dir" and command != "view":
            raise ToolParameterError("path", path, f"The path {path} is a directory and only the `view` command can be used on directories.")
        
        return path_type

    def _is_overwritable_scaffold(self, session: BaseSession, path: str) -> bool:
        """允许系统标记为 scaffold 的文件被 `create` 覆盖。"""
        if not session.is_file(path):
            return False
        try:
            content = session.read_file(path)
        except Exception:
            return False
        if OVERWRITABLE_SCAFFOLD_MARKER in content:
            return True
        if not content.strip():
            return True
        try:
            payload = json.loads(content)
        except Exception:
            return False
        return isinstance(payload, dict) and bool(payload.get("__placeholder__"))

    def _view(
        self,
        session: BaseSession,
        path: str,
        view_range: list[int],
        path_type: Literal["file", "dir", "not_exist"],
    ) -> tuple[str, dict[str, Any]]:
        """查看文件或目录"""
        # 再次检查路径类型，确保判断正确（防止 path_type 误判）
        if path_type == "dir" or session.is_directory(path):
            if view_range:
                raise ToolParameterError("view_range", view_range, "The `view_range` parameter is not allowed for directories.")
            
            # 列出目录内容（最多 2 层）
            result = session.exec_bash(f"find -L {path} -maxdepth 2 -not -path '*/\\.*' | head -500 | sort")
            output = result.get("stdout", "")
            output = maybe_truncate(output, max_size=MAX_OUTPUT_SIZE, notice=DIRECTORY_TRUNCATED_NOTICE)
            
            return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{output}", {}
        
        # 读取文件
        content = session.read_file(path)
        init_line = 1
        
        # 处理 view_range
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolParameterError("view_range", view_range, "It should be a list of two integers.")
            
            lines = content.rstrip("\n").split("\n")
            n_lines = len(lines)
            start, end = view_range
            
            if start < 1 or start > n_lines:
                raise ToolParameterError("view_range", view_range, f"Start line {start} is out of range [1, {n_lines}].")
            if end != -1:
                if end < start:
                    raise ToolParameterError("view_range", view_range, f"End line {end} should be >= start line {start}.")
                if end > n_lines:
                    raise ToolParameterError("view_range", view_range, f"End line {end} exceeds file length {n_lines}.")
            
            if end == -1:
                content = "\n".join(lines[start - 1:])
            else:
                content = "\n".join(lines[start - 1:end])
            init_line = start
        
        return self._format_output(content, path, init_line), {}

    def _create(self, session: BaseSession, path: str, file_text: str) -> tuple[str, dict[str, Any]]:
        """创建文件"""
        previous_content = None
        if session.is_file(path):
            previous_content = session.read_file(path)
        session.write_file(path, file_text)
        history = self._file_history.setdefault(path, [])
        if previous_content is not None:
            history.append((previous_content, "utf-8"))
            return f"Scaffold file overwritten successfully at: {path}", {}
        self._file_history[path] = [(file_text, "utf-8")]
        return f"File created successfully at: {path}", {}

    def _strip_view_number_prefixes(self, text: str) -> str:
        """移除 `view` / `cat -n` 输出中的行号前缀。"""
        lines = text.split("\n")
        normalized = [VIEW_NUMBER_PREFIX_RE.sub("", line) for line in lines]
        return "\n".join(normalized)

    def _candidate_replacements(self, old_str: str, new_str: str) -> list[tuple[str, str]]:
        """为 str_replace 生成一组更稳健的匹配候选。"""
        candidates: list[tuple[str, str]] = []

        def add_candidate(old_value: str, new_value: str) -> None:
            pair = (old_value, new_value)
            if not old_value:
                return
            if pair not in candidates:
                candidates.append(pair)

        add_candidate(old_str, new_str)

        stripped_old = old_str.strip()
        stripped_new = new_str.strip()
        if stripped_old:
            add_candidate(stripped_old, stripped_new)

        denumbered_old = self._strip_view_number_prefixes(old_str)
        denumbered_new = self._strip_view_number_prefixes(new_str)
        if denumbered_old != old_str:
            add_candidate(denumbered_old, denumbered_new)
            stripped_denumbered_old = denumbered_old.strip()
            stripped_denumbered_new = denumbered_new.strip()
            if stripped_denumbered_old:
                add_candidate(stripped_denumbered_old, stripped_denumbered_new)

        return candidates

    def _str_replace(
        self,
        session: BaseSession,
        path: str,
        old_str: str,
        new_str: str,
    ) -> tuple[str, dict[str, Any]]:
        """替换字符串"""
        if new_str == old_str:
            raise ToolParameterError("new_str", new_str, "No replacement was performed. `new_str` and `old_str` must be different.")
        
        content = session.read_file(path)

        matches: list[re.Match[str]] = []
        selected_old = old_str
        selected_new = new_str

        for candidate_old, candidate_new in self._candidate_replacements(old_str, new_str):
            pattern = re.escape(candidate_old)
            candidate_matches = list(re.finditer(pattern, content))
            if not candidate_matches:
                continue
            if candidate_old == candidate_new:
                raise ToolParameterError(
                    "new_str",
                    new_str,
                    "No replacement was performed. `new_str` and `old_str` must be different.",
                )
            selected_old = candidate_old
            selected_new = candidate_new
            matches = candidate_matches
            break

        if not matches:
            raise ToolError(f"No replacement was performed, old_str did not appear verbatim in {path}.")

        if len(matches) > 1:
            # 计算行号
            line_numbers = sorted(set(content.count("\n", 0, m.start()) + 1 for m in matches))
            raise ToolError(f"No replacement was performed. Multiple occurrences of old_str in lines {line_numbers}. Please ensure it is unique.")

        # 执行替换
        match = matches[0]
        replacement_line = content.count("\n", 0, match.start()) + 1
        new_content = content[:match.start()] + selected_new + content[match.end():]
        
        # 保存历史并写入
        if path not in self._file_history:
            self._file_history[path] = []
        self._file_history[path].append((content, "utf-8"))
        session.write_file(path, new_content)
        
        # 创建代码片段
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + selected_new.count("\n") + 1
        snippet = "\n".join(new_content.split("\n")[start_line:end_line + 1])
        
        msg = f"The file {path} has been edited. "
        msg += self._format_output(snippet, f"a snippet of {path}", start_line + 1)
        msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        
        return msg, {}

    def _insert(
        self,
        session: BaseSession,
        path: str,
        insert_line: int,
        new_str: str,
    ) -> tuple[str, dict[str, Any]]:
        """插入内容"""
        content = session.read_file(path)
        lines = content.rstrip("\n").split("\n")
        n_lines = len(lines)
        
        if insert_line < 0 or insert_line > n_lines:
            raise ToolParameterError("insert_line", insert_line, f"It should be within the range [0, {n_lines}]")
        
        # 插入新行
        new_lines = new_str.split("\n")
        result_lines = lines[:insert_line] + new_lines + lines[insert_line:]
        new_content = "\n".join(result_lines)
        
        # 保存历史并写入
        if path not in self._file_history:
            self._file_history[path] = []
        self._file_history[path].append((content, "utf-8"))
        session.write_file(path, new_content)
        
        # 创建代码片段
        start_line = max(0, insert_line - SNIPPET_LINES + 1)
        end_line = insert_line + SNIPPET_LINES + 1
        snippet_lines = lines[start_line:insert_line] + new_lines + lines[insert_line:end_line]
        snippet = "\n".join(snippet_lines)
        
        msg = f"The file {path} has been edited. "
        msg += self._format_output(snippet, "a snippet of the edited file", start_line + 1)
        msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
        
        return msg, {}

    def _undo_edit(self, session: BaseSession, path: str) -> tuple[str, dict[str, Any]]:
        """撤销编辑"""
        if path not in self._file_history or not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")
        
        old_content, old_encoding = self._file_history[path].pop()
        session.write_file(path, old_content, old_encoding)
        
        return f"Last edit to {path} undone successfully. {self._format_output(old_content, path)}", {}

    def _format_output(self, content: str, descriptor: str, init_line: int = 1) -> str:
        """格式化输出（添加行号）"""
        content = maybe_truncate(content, max_size=MAX_OUTPUT_SIZE)
        numbered_lines = [
            f"{i + init_line:6}\t{line}"
            for i, line in enumerate(content.split("\n"))
        ]
        return f"Here's the result of running `cat -n` on {descriptor}:\n" + "\n".join(numbered_lines) + "\n"
