@echo off
REM Local Ollama dev server with the on-disk prompt cache enabled.
REM
REM The prompt cache lets a conversation survive an unload, a restart, or a
REM taskkill: the slot state is mirrored to disk after every completion and
REM read back when the model loads again.
REM
REM Build what this runs first:
REM   go build -ldflags="-s -w" -o build\ollama-dev.exe .
REM   cmake --build build --config Release --target ollama-llama-server-local
REM (the second one is only needed after a change under llama\, and it carries
REM  llama\compat\002-server-slot-checkpoints.patch, without which a restored
REM  hybrid model re-processes the whole prompt on the next turn)

setlocal

REM --- prompt cache -------------------------------------------------------
REM Where the slot states live. Keep it off %TEMP%, which gets cleaned.
if "%OLLAMA_PROMPT_CACHE_DIR%"=="" set OLLAMA_PROMPT_CACHE_DIR=%LOCALAPPDATA%\Ollama\prompt-cache
if not exist "%OLLAMA_PROMPT_CACHE_DIR%" mkdir "%OLLAMA_PROMPT_CACHE_DIR%"

REM Host memory for displaced prompts, in MiB. Defaults to 0 whenever the
REM cache is on disk; raise it only if you have RAM to spare.
if "%OLLAMA_PROMPT_CACHE_RAM%"=="" set OLLAMA_PROMPT_CACHE_RAM=0

REM Unload an idle model even under an infinite keep-alive, so its memory goes
REM back to the system. The state is on disk, so the next request pays a
REM reload instead of a re-prefill. Empty or 0 disables this.
if "%OLLAMA_PROMPT_CACHE_IDLE_UNLOAD%"=="" set OLLAMA_PROMPT_CACHE_IDLE_UNLOAD=10m

REM --- memory -------------------------------------------------------------
REM One slot: the hybrid architectures do not support parallel requests, and a
REM second slot doubles the context allocation for nothing.
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1
REM Deprecated: set OLLAMA_NUM_BATCH=8192
set OLLAMA_NUM_UBATCH=2048
set LLAMA_ATTN_ROT_DISABLE=1

REM --- model -------------------------------------------------------------
if "%OLLAMA_KEEP_ALIVE%"=="" set OLLAMA_KEEP_ALIVE=-1
if "%OLLAMA_CONTEXT_LENGTH%"=="" set OLLAMA_CONTEXT_LENGTH=262144
if "%OLLAMA_FLASH_ATTENTION%"=="" set OLLAMA_FLASH_ATTENTION=1
if "%OLLAMA_CPU_MOE%"=="" set OLLAMA_CPU_MOE=1

REM Nemotron-H hybrids only have a KV cache on their few attention layers (8 of
REM 88 on Bolt: ~0.5 GiB at 128k in q8_0), so q4_0 saves ~250 MiB for a real
REM quality cost. The experts live in host memory (llm/host_experts.go), the
REM VRAM that is left becomes a GPU LRU of hot experts: OLLAMA_MOE_CACHE_MIB
REM overrides the automatic size, OLLAMA_CPU_MOE=0 turns the whole thing off.
if "%OLLAMA_KV_CACHE_TYPE%"=="" set OLLAMA_KV_CACHE_TYPE=q4_0
REM --- server -------------------------------------------------------------
if "%OLLAMA_HOST%"=="" set OLLAMA_HOST=127.0.0.1:11434

set OLLAMA_EXE=%~dp0build\ollama-dev.exe
if not exist "%OLLAMA_EXE%" (
    echo Build it first: go build -ldflags="-s -w" -o build\ollama-dev.exe .
    exit /b 1
)

echo prompt cache : %OLLAMA_PROMPT_CACHE_DIR%
echo cache RAM    : %OLLAMA_PROMPT_CACHE_RAM% MiB
echo idle unload  : %OLLAMA_PROMPT_CACHE_IDLE_UNLOAD%
echo context      : %OLLAMA_CONTEXT_LENGTH%  parallel: %OLLAMA_NUM_PARALLEL%
echo host         : %OLLAMA_HOST%
echo.

"%OLLAMA_EXE%" serve
