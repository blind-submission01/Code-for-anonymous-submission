#!/usr/bin/env bash

# 获取当前脚本所在目录
bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 确保bin目录下的脚本可执行
chmod +x "$bundle_dir"/bin/*

# 将bin目录添加到PATH（这一步在ToolHandler._install_commands中已经完成，但再次确认）
export PATH="$bundle_dir/bin":$PATH

# 确保脚本使用LF行尾，避免Windows CRLF问题
if command -v dos2unix &> /dev/null; then
    dos2unix "$bundle_dir"/bin/*
else
    # 如果没有dos2unix，使用sed替代（在大多数Linux系统上可用）
    sed -i 's/\r$//' "$bundle_dir"/bin/*
fi

# 设置必要的环境变量（如果有的话）
# 注意：实际API密钥应通过registry_variables或env_variables配置

echo "Guidance 工具安装完成"