# 部署说明

## 公网工作台

- 服务器：`124.174.16.151`
- 访问端口：`8233`
- 容器名：`long-novel-gpt`
- 镜像名：`long-novel-gpt:amd64`

## 必须持久化的数据

v2 工作台的模型配置、系统参数、小说数据都写在容器内 `/app/data`。生产部署必须挂载到宿主机目录，否则重建容器后会出现模型配置为空、历史小说丢失的问题。

当前服务器固定使用：

```bash
-v /opt/lng/data:/app/data
```

## 推荐启动命令

```bash
docker rm -f long-novel-gpt >/dev/null 2>&1 || true
docker run -d \
  --name long-novel-gpt \
  --restart unless-stopped \
  -p 8233:80 \
  -v /opt/lng/data:/app/data \
  long-novel-gpt:amd64
```

## 部署后检查

```bash
curl -fsS http://124.174.16.151:8233/api/health
curl -fsS http://124.174.16.151:8233/api/v2/models
docker ps --filter name=long-novel-gpt
```

`/api/v2/models` 必须能看到已配置模型和 `active_id`，不能是空数组。
