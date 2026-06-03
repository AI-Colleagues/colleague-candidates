# Design Document

## For WeChat Medical Reminder

- **Version:** 0.1
- **Author:** ShaojieJiang
- **Date:** 2026-02-23
- **Status:** Approved

---

## Overview

The WeChat Medical Reminder system is built on the Orcheo workflow platform and consists of two workflows that share two MongoDB collections. The **Message Handler Workflow** processes incoming WeChat messages via WeCom Customer Service webhooks. An AI agent (AgentNode) interprets user intent and manages registration, deregistration, and status reporting through MongoDB tool calls. The **Daily Reminder Workflow** runs on a cron schedule at 9:00 AM Asia/Shanghai, retrieves all active users, and sends personalised reminders via a ForLoopNode iteration. Each sent reminder is also persisted to the LangGraph graph store so the agent can reference reminder context when the user replies.

Both workflows communicate with external WeChat users exclusively through the WeCom Customer Service (微信客服) channel, requiring no WeCom internal account on the user's side.

## Components

- **Message Handler Workflow (webhook-triggered)**
  - Receives WeCom callback events, validates signatures, syncs customer service messages, looks up user registration, and routes them to an AI agent for processing
  - Key nodes: WebhookTriggerNode, WeComEventsParserNode, WeComAccessTokenNode, WeComCustomerServiceSyncNode, MongoDBFindNode, PrepareAgentContextNode (custom), AgentNode, AgentReplyExtractorNode, WeComCustomerServiceSendNode

- **Daily Reminder Workflow (cron-triggered)**
  - Fetches all active registered users and sends personalised reminders via ForLoopNode iteration, persisting each reminder to the graph store for agent conversation history
  - Key nodes: CronTriggerNode, WeComAccessTokenNode, MongoDBFindNode, ForLoopNode, PrepareMessageNode (custom), WeComCustomerServiceSendNode, GraphStoreAppendMessageNode

- **DB Setup Workflow (manual trigger)**
  - One-time (idempotent) admin workflow to create the required MongoDB database, collections, and indexes
  - Key nodes: MongoDBNode (create_index)

- **MongoDB (shared storage)**
  - `registered_users` collection: user profiles with reminder items
  - `user_records` collection: daily status reports

- **Orcheo Vault (secrets)**
  - WeCom credentials, MongoDB connection string, LLM API key

## Request Flows

### Flow 1: User Registration

1. WeChat user sends a message to the WeCom Customer Service account with their reminder items and phone number (e.g., "我想注册提醒，每天吃降压药、测血压，手机号13800138000")
2. WeCom sends a callback to the Orcheo webhook endpoint
3. `WeComEventsParserNode` validates the signature and parses the callback
4. `WeComAccessTokenNode` fetches/caches the access token
5. `WeComCustomerServiceSyncNode` syncs the user's message and extracts `external_userid` and `open_kf_id`
6. `AgentNode` receives the message with conversation history and context (external_userid, open_kf_id)
7. Agent calls `mongodb_find` on `registered_users` to check if user already exists
8. If user does not exist: Agent calls `mongodb_update_one` with `upsert: true` to create the registration
9. Agent generates a confirmation reply summarising the registered items
10. `ExtractAgentReplyNode` extracts the agent's reply text
11. `WeComCustomerServiceSendNode` sends the reply back to the user

### Flow 2: User Deregistration

1. Registered user sends a message expressing deregistration intent (e.g., "我不想再接收提醒了")
2. Steps 2-6 same as Flow 1
3. Agent detects deregistration intent and replies asking for confirmation (e.g., "确定要取消提醒服务吗？请回复'确认'")
4. Reply is sent to user via `WeComCustomerServiceSendNode`
5. User sends confirmation message (e.g., "确认")
6. A new webhook invocation processes this message; the AgentNode sees the prior conversation history (via LangGraph checkpointing) and recognises the confirmation
7. Agent calls `mongodb_update_one` on `registered_users` to set `status: "inactive"`
8. Agent generates a deregistration confirmation reply
9. Reply is sent to user

### Flow 3: Daily Reminder (Cron)

1. `CronTriggerNode` fires at 09:00 Asia/Shanghai daily
2. `WeComAccessTokenNode` fetches/caches the access token
3. `MongoDBFindNode` queries `registered_users` with filter `{"status": "active"}` (also requires `external_userid` and `open_kf_id` to exist and be non-empty)
4. `ForLoopNode` (`for_each_user`) receives the user list and exposes one user per iteration via `current_item`
5. Conditional edge checks `for_each_user.done`: if falsy, proceed to loop body; if truthy, go to END
6. `PrepareMessageNode` reads the current user from `for_each_user.current_item` and formats a personalised reminder from their `reminder_items` and `external_username`
7. `WeComCustomerServiceSendNode` (`send_reminder`) delivers the message (`raise_on_error=False` so one failure doesn't abort the batch)
8. `GraphStoreAppendMessageNode` (`persist_reminder_history`) persists the sent reminder to the LangGraph graph store under key `wecom_cs:{open_kf_id}:{external_userid}`, making it available as conversation history when the user later replies to the Message Handler agent
9. Loop back to `for_each_user` for the next iteration; continues until all users have been processed

### Flow 4: User Status Report

1. Registered user sends a status report message (e.g., "今天药已经吃了，血压130/85")
2. Steps 2-6 same as Flow 1
3. Agent detects status-report intent, extracts structured data from the message
4. Agent calls `mongodb_update_one` on `user_records` with `upsert: true`, using filter `{"external_userid": "<id>", "open_kf_id": "<id>", "record_date": "<today>"}` and the extracted status data
5. Agent generates a confirmation reply (e.g., "已记录今日状态：降压药已服用，血压130/85")
6. Reply is sent to user via `WeComCustomerServiceSendNode`

### Flow 5: Database Setup (Admin)

1. Admin triggers the DB Setup workflow manually via `orcheo workflow run`
2. `create_registered_users_index` node (MongoDBNode, operation: `create_index`) creates a compound ascending unique index on `{external_userid, open_kf_id}` in the `registered_users` collection — implicitly creating the database and collection if they don't exist. The compound key ensures uniqueness per user-per-KF-account rather than globally per user
3. `create_user_records_index` node (MongoDBNode, operation: `create_index`) creates a compound ascending unique index on `{external_userid, open_kf_id, record_date}` in the `user_records` collection — implicitly creating the collection if it doesn't exist
4. Both operations are idempotent: re-running the workflow is safe

## API Contracts

### Webhook Endpoint (Message Handler Workflow)

```
GET/POST /api/workflows/{workflow_id}/triggers/webhook?preserve_raw_body=true

WeCom sends callbacks to this endpoint.
- GET with echostr: URL verification (handled by WeComEventsParserNode)
- POST with encrypted XML: Message callbacks
```

### Orcheo Vault Credentials

```
wecom_app_secret_medical_reminder - WeCom app secret for access token
wecom_corp_id                     - WeCom corp ID
wecom_token                       - Callback token for signature validation
wecom_encoding_aes_key            - AES key for callback decryption
mdb_connection_string             - MongoDB connection string
openai_api_key                    - OpenAI API key for the LLM agent
```

### Workflow Configurable Inputs

These values should be specified in the workflow's config file (e.g., `workflow.json`):

```json
{
  "configurable": {
    "reminder_database": "<MongoDB database name>",
    "registered_users_collection": "registered_users",
    "user_records_collection": "user_records"
  }
}
```

## Data Models / Schemas

### registered_users Collection

| Field | Type | Description |
|-------|------|-------------|
| external_userid | string | WeChat external user ID from WeCom Customer Service sync |
| open_kf_id | string | WeCom Customer Service account ID (required for sending messages) |
| phone_number | string | User's phone number |
| reminder_items | string[] | List of personalised reminder items (e.g., ["吃降压药", "测血压"]) |
| status | string | `"active"` or `"inactive"` (soft-delete for deregistration) |
| registered_at | string | ISO 8601 datetime of initial registration |
| updated_at | string | ISO 8601 datetime of last update |

```json
{
  "external_userid": "wmXXXXXXXXXX",
  "open_kf_id": "wkXXXXXXXXXX",
  "phone_number": "13800138000",
  "reminder_items": ["吃降压药", "测血压", "30分钟步行"],
  "status": "active",
  "registered_at": "2026-02-23T10:30:00+08:00",
  "updated_at": "2026-02-23T10:30:00+08:00"
}
```

### user_records Collection

| Field | Type | Description |
|-------|------|-------------|
| external_userid | string | WeChat external user ID (foreign key to registered_users) |
| open_kf_id | string | WeCom Customer Service account ID (ensures per-account data isolation) |
| record_date | string | Date of the record in `YYYY-MM-DD` format |
| raw_text | string | Original user message text |
| items_status | object | Structured status per reminder item |
| recorded_at | string | ISO 8601 datetime when the record was stored |

```json
{
  "external_userid": "wmXXXXXXXXXX",
  "open_kf_id": "wkXXXXXXXXXX",
  "record_date": "2026-02-23",
  "raw_text": "今天药已经吃了，血压130/85，没有走路",
  "items_status": {
    "吃降压药": "已完成",
    "测血压": "130/85",
    "30分钟步行": "未完成"
  },
  "recorded_at": "2026-02-23T11:15:00+08:00"
}
```

## Agent System Prompt (Message Handler)

The AgentNode in the Message Handler Workflow uses a dynamically-built system prompt assembled by `PrepareAgentContextNode`. The context section resolves all variables at runtime — template placeholders below show where dynamic values are injected:

```
你是一个微信健康提醒助理。你通过自然语言对话帮助用户管理健康提醒注册和每日状态记录。

上下文（以下为实际值，请在工具调用中直接使用）：
- external_userid: <value>{resolved from wecom_cs_sync}</value>
- external_username: <value>{resolved from wecom_cs_sync}</value>
- open_kf_id: <value>{resolved from wecom_cs_sync}</value>
- 当前日期: 2026-02-23            ← concrete ISO date
- 当前时间: 2026-02-23T09:00:00+08:00  ← concrete ISO datetime
- 用户注册状态: active/inactive/未注册  ← from lookup_user result
- 提醒项目: ["吃降压药", "测血压"]     ← only if registered

MongoDB 配置：
- database: {configurable.reminder_database}
- registered_users collection: {configurable.registered_users_collection}
- user_records collection: {configurable.user_records_collection}

工具：
- mongodb_find：查询文档（支持 filter/sort/limit）
- mongodb_update_one：更新或插入文档（支持 filter/update/options）

你负责以下任务：

1. 用户注册：
   - 从用户消息中提取提醒项目列表和手机号
   - 先用 mongodb_find 检查 registered_users 中是否已存在该
     external_userid + open_kf_id 组合
   - 如不存在，用 mongodb_update_one (upsert: true) 创建记录，包含：
     external_userid, external_username, open_kf_id, phone_number,
     reminder_items, status: "active", registered_at, updated_at
   - 如已存在且 status 为 active，提示用户已注册
   - 确认注册并列出已保存的提醒项目

2. 用户注销：
   - 检测用户注销意图时，先回复确认请求（如"确定要取消提醒服务吗？请回复'确认'"）
   - 收到确认后，用 mongodb_update_one 将 status 设为 "inactive"，更新 updated_at
   - 确认注销完成

3. 状态报告：
   - 只有注册状态为 active 的用户才能报告状态
   - 如果用户注册状态为 inactive 或未注册，拒绝记录并提示用户需要先注册
   - 用户报告每日健康状态时，提取各提醒项目的完成情况
   - 用 mongodb_update_one (upsert: true) 在 user_records 中创建记录：
     filter: {"external_userid": "<id>", "open_kf_id": "<id>", "record_date": "<YYYY-MM-DD>"}
     update: {"$set": {raw_text, items_status, recorded_at, external_userid,
     open_kf_id, record_date}}
   - 确认已记录并总结状态

4. 其他消息：
   - 如用户消息不属于以上类别，友好地介绍你的功能并引导用户

指南：
- 所有回复使用中文纯文本，不要输出 JSON 或 Markdown
- 对破坏性操作（注销）必须先确认
- 手机号格式验证：中国大陆手机号（11位，1开头）
- 提醒项目应以简洁短语存储
- 在合适的场景下使用用户的 external_username 称呼用户，如问候、确认操作等
- 所有 mongodb 查询和更新必须同时使用 external_userid 和 open_kf_id 作为筛选条件，
  确保不同客服账号的数据互相独立
```

## Node & Edge Configuration

### Message Handler Workflow Nodes

| Node Name | Node Type | Key Configuration |
|-----------|-----------|-------------------|
| `webhook_trigger` | WebhookTriggerNode | `allowed_methods: ["GET", "POST"]` |
| `wecom_events_parser` | WeComEventsParserNode | (defaults) |
| `get_cs_access_token` | WeComAccessTokenNode | `app_secret: [[wecom_app_secret_medical_reminder]]` |
| `wecom_cs_sync` | WeComCustomerServiceSyncNode | (defaults) |
| `lookup_user` | MongoDBFindNode | `filter: {external_userid, open_kf_id}`, `limit: 1` — looks up current registration status |
| `prepare_agent_context` | PrepareAgentContextNode | Custom TaskNode: builds the system prompt with resolved context values (user IDs, date, registration status, MongoDB config) |
| `agent` | AgentNode | `ai_model: "openai:gpt-4o-mini"`, `predefined_tools: ["mongodb_find", "mongodb_update_one"]`, `use_graph_chat_history: true`, `history_key_candidates: ["wecom_cs:{open_kf_id}:{external_userid}"]` |
| `extract_agent_reply` | AgentReplyExtractorNode | `fallback_message: "抱歉，处理您的请求时遇到了问题，请稍后再试。"` |
| `send_cs_reply` | WeComCustomerServiceSendNode | `message: {{extract_agent_reply.agent_reply}}` |

### Message Handler Workflow Edges

| From | To | Condition |
|------|----|-----------|
| START | `webhook_trigger` | (entry point) |
| `webhook_trigger` | `wecom_events_parser` | (unconditional) |
| `wecom_events_parser` | END | If `immediate_response` is truthy |
| `wecom_events_parser` | `get_cs_access_token` | Otherwise |
| `get_cs_access_token` | `wecom_cs_sync` | (unconditional) |
| `wecom_cs_sync` | END | If `should_process` is falsy |
| `wecom_cs_sync` | `lookup_user` | Otherwise |
| `lookup_user` | `prepare_agent_context` | (unconditional) |
| `prepare_agent_context` | `agent` | (unconditional) |
| `agent` | `extract_agent_reply` | (unconditional) |
| `extract_agent_reply` | `send_cs_reply` | (unconditional) |
| `send_cs_reply` | END | (unconditional) |

### Daily Reminder Workflow Nodes

| Node Name | Node Type | Key Configuration |
|-----------|-----------|-------------------|
| `cron_trigger` | CronTriggerNode | `expression: "0 9 * * *"`, `timezone: "Asia/Shanghai"` |
| `get_access_token` | WeComAccessTokenNode | `app_secret: [[wecom_app_secret_medical_reminder]]` |
| `find_active_users` | MongoDBFindNode | `collection: "registered_users"`, `filter: {"status": "active", "external_userid": {"$exists": true, "$ne": ""}, "open_kf_id": {"$exists": true, "$ne": ""}}` |
| `for_each_user` | ForLoopNode | `items: {{find_active_users.data}}` — iterates over users, exposes `current_item` and `done` |
| `prepare_message` | PrepareMessageNode | Custom TaskNode: reads `for_each_user.current_item`, formats personalised message from `reminder_items` and `external_username` |
| `send_reminder` | WeComCustomerServiceSendNode | `raise_on_error: false` — sends to current user; failures don't abort the batch |
| `persist_reminder_history` | GraphStoreAppendMessageNode | `key: "wecom_cs:{open_kf_id}:{external_userid}"` — persists the reminder as conversation history for the Message Handler agent |

### Daily Reminder Workflow Edges

| From | To | Condition |
|------|----|-----------|
| START | `cron_trigger` | (entry point) |
| `cron_trigger` | `get_access_token` | (unconditional) |
| `get_access_token` | `find_active_users` | (unconditional) |
| `find_active_users` | `for_each_user` | (unconditional) |
| `for_each_user` | `prepare_message` | IfElse: `for_each_user.done` is falsy (more users) |
| `for_each_user` | END | IfElse: `for_each_user.done` is truthy (all done) |
| `prepare_message` | `send_reminder` | (unconditional) |
| `send_reminder` | `persist_reminder_history` | (unconditional) |
| `persist_reminder_history` | `for_each_user` | (unconditional, loop back) |

### DB Setup Workflow Nodes

| Node Name | Node Type | Key Configuration |
|-----------|-----------|-------------------|
| `create_registered_users_index` | MongoDBNode | `operation: "create_index"`, `collection: "registered_users"`, `keys: {"external_userid": 1, "open_kf_id": 1}`, `kwargs: {"name": "idx_userid_kfid", "unique": true}` |
| `create_user_records_index` | MongoDBNode | `operation: "create_index"`, `collection: "user_records"`, `keys: {"external_userid": 1, "open_kf_id": 1, "record_date": 1}`, `kwargs: {"name": "idx_userid_kfid_date", "unique": true}` |

### DB Setup Workflow Edges

| From | To | Condition |
|------|----|-----------|
| START | `create_registered_users_index` | (entry point) |
| `create_registered_users_index` | `create_user_records_index` | (unconditional) |
| `create_user_records_index` | END | (unconditional) |

## Security Considerations

- **WeCom signature validation**: All incoming callbacks are verified by WeComEventsParserNode using the Token and EncodingAESKey
- **Secrets management**: All sensitive credentials (WeCom secrets, MongoDB connection string, OpenAI API key) stored in Orcheo Vault using `[[secret_name]]` syntax
- **Phone number privacy**: Phone numbers stored in MongoDB should be treated as PII; access restricted to the workflow
- **Input validation**: Agent system prompt enforces phone number format validation; MongoDB operations use parameterised filters (no injection risk)
- **Soft-delete for deregistration**: User data is not permanently deleted, preserving audit trail

## Performance Considerations

- **Cron batch size**: For large user bases, the ForLoopNode iteration may hit WeCom API rate limits. Consider adding a `DelayNode` between iterations for throttling.
- **MongoDB indexing**: Compound unique index on `registered_users.{external_userid, open_kf_id}` enforces per-user-per-KF-account uniqueness; compound unique index on `user_records.{external_userid, open_kf_id, record_date}` for per-account-isolated daily lookups
- **LLM latency**: GPT-4o-mini typically responds within 2-5 seconds; total message handler latency should stay under 10 seconds
- **Access token caching**: WeComAccessTokenNode caches tokens internally, avoiding redundant API calls

## Testing Strategy

- **Unit tests**: Custom TaskNode logic (PrepareMessageNode, PrepareAgentContextNode)
- **Integration tests**: End-to-end workflow runs using `orcheo workflow run` with mock WeCom payloads
- **Agent behaviour tests**: Verify agent correctly handles registration, deregistration, and status report with sample Chinese messages
- **Cron workflow tests**: Verify iteration logic with mock MongoDB data (0 users, 1 user, multiple users)
- **Manual QA checklist**:
  - Register a new user via WeChat message
  - Attempt duplicate registration (should inform user)
  - Deregister with confirmation flow
  - Verify cron reminder delivery at 9 AM
  - Submit a status report and verify MongoDB record

## Rollout Plan

1. Phase 1: Deploy to local Orcheo instance; test with mock WeCom callbacks via `orcheo workflow run`
2. Phase 2: Connect to WeCom Customer Service sandbox; test with internal team
3. Phase 3: Production deployment with real WeCom Customer Service account; gradual user onboarding

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-23 | ShaojieJiang | Initial draft |
| 2026-02-23 | ShaojieJiang | Added DB Setup workflow (Flow 5) for admin-driven MongoDB provisioning |
| 2026-03-02 | ShaojieJiang | Updated for ForLoopNode refactor: replaced SubWorkflow/While loop with ForLoopNode; added persist_reminder_history, lookup_user, prepare_agent_context nodes; updated agent system prompt example; fixed registered_users index to compound {external_userid, open_kf_id} |
| 2026-03-02 | ShaojieJiang | Added open_kf_id to user_records filter/index for full per-account isolation; fixed vault secret names; removed stale corp_id configurable; added open_kf_id to user_records data model |
