# [FastAPI 专家]的 101 个 FastAPI 技巧

> 本文档翻译自 [fastapi-tips](https://github.com/Kludex/fastapi-tips)。

这个仓库包含了 FastAPI 的技巧和窍门。如果你有任何你认为有用的技巧，欢迎提交 issue 或者 pull request。

请考虑在 GitHub 上赞助我以支持我的工作。有了你的支持，我将能够创作更多类似的内容。

[![GitHub 赞助](https://img.shields.io/badge/Sponsor%20me%20on-GitHub-%23EA4AAA)](https://github.com/sponsors/Kludex)

> [!TIP]
> 记得**关注这个仓库**以接收新技巧的通知。

## 1. 安装 `uvloop` 和 `httptools`

默认情况下，[Uvicorn][uvicorn] 不包含 `uvloop` 和 `httptools`，它们比默认的 asyncio 事件循环和 HTTP 解析器更快。你可以使用以下命令安装它们：

```bash
pip install uvloop httptools
```

[Uvicorn][uvicorn] 会在你的环境中安装了它们的情况下自动使用它们。

> [!WARNING]
> `uvloop` 不能在 Windows 上安装。如果你在本地使用 Windows，但在生产环境中使用 Linux，你可以使用一个 [环境标记](https://peps.python.org/pep-0496/) 来在 Windows 上不安装 `uvloop`
> 例如 `uvloop; sys_platform != 'win32'`。

## 2. 小心非异步函数

在 FastAPI 中使用非异步函数时会有性能损失。所以，尽量使用异步函数。
这个性能损失是因为 FastAPI 会调用 [`run_in_threadpool`][run_in_threadpool]，它会使用一个线程池来运行这个函数。

> [!NOTE]
> 在内部，[`run_in_threadpool`][run_in_threadpool] 会使用 [`anyio.to_thread.run_sync`][run_sync] 在线程池中运行这个函数。

> [!TIP]
> 线程池中只有 40 个线程可用。如果你使用了所有的线程，你的应用程序将被阻塞。

要改变线程池中可用的线程数量，你可以使用以下代码：

```py
import anyio
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 100
    yield

app = FastAPI(lifespan=lifespan)
```

你可以在 [AnyIO's 文档][increase-threadpool] 中阅读更多相关信息。

## 3. 使用 `async for` 代替 `while True` 处理 WebSocket

大多数你在网上找到的示例都会使用 `while True` 从 WebSocket 读取消息。

我认为这种不太优雅的写法主要是因为 Starlette 文档很长时间没有展示 `async for` 的用法。

与其使用 `while True`：

```py
from fastapi import FastAPI
from starlette.websockets import WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```

你可以使用 `async for` 语法：

```py
from fastapi import FastAPI
from starlette.websockets import WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    async for data in websocket.iter_text():
        await websocket.send_text(f"Message text was: {data}")
```

你可以在 [Starlette 文档][websockets-iter-data] 中阅读更多相关信息。

## 4. 忽略 `WebSocketDisconnect` 异常

如果你使用 `while True` 语法，你需要捕获 `WebSocketDisconnect` 异常。
而使用 `async for` 语法会自动捕获该异常。

```py
from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        pass
```

如果你需要在 WebSocket 断开连接时释放资源，可以使用该异常来处理。

如果你使用的是旧版本的 FastAPI，只有 `receive` 方法会引发 `WebSocketDisconnect` 异常。
`send` 方法不会引发该异常。在最新版本中，所有方法都会引发该异常。
在这种情况下，你需要将 `send` 方法放在 `try` 块中。

## 5. 使用 HTTPX 的 `AsyncClient` 代替 `TestClient`

由于你的应用程序中使用了 `async` 函数，使用 HTTPX 的 `AsyncClient` 会比使用 Starlette 的 `TestClient` 更加方便。

```py
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# 使用 TestClient
from starlette.testclient import TestClient

client = TestClient(app)
response = client.get("/")
assert response.status_code == 200
assert response.json() == {"Hello": "World"}

# 使用 AsyncClient
import anyio
from httpx import AsyncClient, ASGITransport


async def main():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Hello": "World"}


anyio.run(main)
```

如果你使用生命周期事件（`on_startup`、`on_shutdown` 或 `lifespan` 参数），可以使用 [`asgi-lifespan`][asgi-lifespan] 包来运行这些事件。

```py
from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("Starting app")
    yield
    print("Stopping app")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


async def main():
    async with LifespanManager(app) as manager:
        async with AsyncClient(transport=ASGITransport(app=manager.app)) as client:
            response = await client.get("/")
            assert response.status_code == 200
            assert response.json() == {"Hello": "World"}


anyio.run(main)
```

> [!NOTE]
> 请考虑通过 GitHub 赞助支持 [`asgi-lifespan`][asgi-lifespan] 的创建者 [Florimond Manca][florimondmanca]。

## 6. 使用生命周期状态代替 `app.state`

不久前，FastAPI 开始支持 [生命周期状态]，它定义了一种标准的方法来管理在启动时需要创建的对象，并在请求-响应周期中使用这些对象。

不再推荐使用 `app.state`。你应该使用 [生命周期状态] 代替。

使用 `app.state` 时，你可能会这样做：

```py
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from httpx import AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with AsyncClient(app=app) as client:
        app.state.client = client
        yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root(request: Request):
    client = request.app.state.client
    response = await client.get("/")
    return response.json()
```

使用生命周期状态时，你可以这样做：

```py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict, cast

from fastapi import FastAPI, Request
from httpx import AsyncClient


class State(TypedDict):
    client: AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    async with AsyncClient(app=app) as client:
        yield {"client": client}


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root(request: Request) -> dict[str, Any]:
    client = cast(AsyncClient, request.state.client)
    response = await client.get("/")
    return response.json()
```

## 7. 启用 AsyncIO 调试模式

如果你想找到阻塞事件循环的端点，可以启用 AsyncIO 调试模式。

启用后，当一个任务执行时间超过 100 毫秒时，Python 会打印警告信息。

使用以下命令运行代码：`PYTHONASYNCIODEBUG=1 python main.py`：

```py
import os
import time

import uvicorn
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def read_root():
    time.sleep(1)  # 阻塞调用
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, loop="uvloop")
```

如果你调用该端点，你将看到以下消息：

```bash
INFO:     Started server process [19319]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:50036 - "GET / HTTP/1.1" 200 OK
Executing <Task finished name='Task-3' coro=<RequestResponseCycle.run_asgi() done, defined at /uvicorn/uvicorn/protocols/http/httptools_impl.py:408> result=None created at /uvicorn/uvicorn/protocols/http/httptools_impl.py:291> took 1.009 seconds
```

你可以在 [官方文档](https://docs.python.org/3/library/asyncio-dev.html#debug-mode) 中阅读更多相关信息。

## 8. 实现一个纯 ASGI 中间件代替 `BaseHTTPMiddleware`

[`BaseHTTPMiddleware`][base-http-middleware] 是在 FastAPI 中创建中间件的最简单方法。

> [!NOTE]
> `@app.middleware("http")` 装饰器是 `BaseHTTPMiddleware` 的包装器。

`BaseHTTPMiddleware` 存在一些问题，但大多数问题在最新版本中已修复。
尽管如此，使用它仍然会有性能损失。

为了避免性能损失，你可以实现一个 [纯 ASGI 中间件]。缺点是实现起来更复杂。

查看 Starlette 的文档以了解如何实现 [纯 ASGI 中间件].

## 9. 你的依赖项可能在线程中运行

如果函数是非异步的，并且你将其用作依赖项，它将在一个线程中运行。

在以下示例中，`http_client` 函数将在一个线程中运行：

```py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from httpx import AsyncClient
from fastapi import FastAPI, Request, Depends


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, AsyncClient]]:
    async with AsyncClient() as client:
        yield {"client": client}


app = FastAPI(lifespan=lifespan)


def http_client(request: Request) -> AsyncClient:
    return request.state.client


@app.get("/")
async def read_root(client: AsyncClient = Depends(http_client)):
    return await client.get("/")
```

要在事件循环中运行，你需要将函数改为异步：

```py
# ...

async def http_client(request: Request) -> AsyncClient:
    return request.state.client

# ...
```

作为练习，让我们了解更多关于如何检查运行线程的信息。

你可以使用 `python main.py` 运行以下代码：

```py
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from anyio.to_thread import current_default_thread_limiter
from httpx import AsyncClient
from fastapi import FastAPI, Request, Depends


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, AsyncClient]]:
    async with AsyncClient() as client:
        yield {"client": client}


app = FastAPI(lifespan=lifespan)


# 将此函数改为异步，并重新运行此应用程序。
def http_client(request: Request) -> AsyncClient:
    return request.state.client


@app.get("/")
async def read_root(client: AsyncClient = Depends(http_client)): ...


async def monitor_thread_limiter():
    limiter = current_default_thread_limiter()
    threads_in_use = limiter.borrowed_tokens
    while True:
        if threads_in_use != limiter.borrowed_tokens:
            print(f"Threads in use: {limiter.borrowed_tokens}")
            threads_in_use = limiter.borrowed_tokens
        await anyio.sleep(0)


if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(app="main:app")
    server = uvicorn.Server(config)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(monitor_thread_limiter)
            await server.serve()

    anyio.run(main)
```

如果你调用该端点，你将看到以下消息：

```bash
❯ python main.py
INFO:     Started server process [23966]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Threads in use: 1
INFO:     127.0.0.1:57848 - "GET / HTTP/1.1" 200 OK
Threads in use: 0
```

将 `def http_client` 替换为 `async def http_client` 并重新运行应用程序。
你将不会看到 `Threads in use: 1` 的消息，因为该函数在事件循环中运行。

> [!TIP]
> 你可以使用我构建的 [FastAPI Dependency] 包来明确指定依赖项何时应该在线程中运行。

[uvicorn]: https://www.uvicorn.org/
[run_sync]: https://anyio.readthedocs.io/en/stable/threads.html#running-a-function-in-a-worker-thread
[run_in_threadpool]: https://github.com/encode/starlette/blob/9f16bf5c25e126200701f6e04330864f4a91a898/starlette/concurrency.py#L36-L42
[increase-threadpool]: https://anyio.readthedocs.io/en/stable/threads.html#adjusting-the-default-maximum-worker-thread-count
[websockets-iter-data]: https://www.starlette.io/websockets/#iterating-data
[florimondmanca]: https://github.com/sponsors/florimondmanca
[asgi-lifespan]: https://github.com/florimondmanca/asgi-lifespan
[生命周期状态]: https://asgi.readthedocs.io/en/latest/specs/lifespan.html#lifespan-state
[FastAPI 专家]: https://github.com/Kludex
[base-http-middleware]: https://www.starlette.io/middleware/#basehttpmiddleware
[纯 ASGI 中间件]: https://www.starlette.io/middleware/#pure-asgi-middleware
[FastAPI Dependency]: https://github.com/kludex/fastapi-dependency
