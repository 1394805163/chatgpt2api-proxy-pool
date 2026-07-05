# ChatGPT2API Proxy Pool

这是一个基于 `basketikun/chatgpt2api` 的 Docker 可部署版本，默认从本仓库源码本地构建镜像。

## 快速启动

Windows 用户可以双击：

```text
start-docker.bat
```

也可以手动运行：

```bash
docker compose up -d --build
```

启动后访问：

```text
Web 面板：http://localhost:3000
API 地址：http://localhost:3000/v1
默认 API Key：12345678
```

## 修改 API Key

推荐复制 `.env.example` 为 `.env`，然后修改：

```env
CHATGPT2API_AUTH_KEY=你的新 key
CHATGPT2API_PORT=3000
STORAGE_BACKEND=json
```

`.env` 不要提交到公开仓库。仓库默认值保持为 `12345678`，方便新手直接启动。

## 更新项目

Windows 用户可以双击：

```text
update-docker.bat
```

也可以手动运行：

```bash
git pull
docker compose up -d --build
```

一般不需要手动删除旧容器，Compose 会按新镜像和配置自动重建并替换服务。

## 数据位置

运行数据保存在：

```text
./data
./config.json
```

公开上传前请确认不要提交真实账号、Token、日志、私有 `.env` 或本机真实配置。

旧版完整说明已归档到 [docs/legacy-readme.md](./docs/legacy-readme.md)。
