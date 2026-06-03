# Project Plan

## For WeChat Medical Reminder

- **Version:** 0.1
- **Author:** ShaojieJiang
- **Date:** 2026-02-23
- **Status:** Approved

---

## Overview

Build two Orcheo workflows for a WeChat-based medical reminder system: a webhook-triggered Message Handler for user interactions (registration, deregistration, status reporting) and a cron-triggered Daily Reminder for sending personalised health reminders. Both workflows share `registered_users` and `user_records` MongoDB collections.

**Related Documents:**
- Requirements: [1_requirements.md](./1_requirements.md)
- Design: [2_design.md](./2_design.md)

---

## Milestones

### Milestone 1: Infrastructure & Credentials Setup

**Description:** Set up the Orcheo environment, WeCom Customer Service configuration, MongoDB collections, and Orcheo vault secrets. Success criteria: all credentials verified and MongoDB collections accessible.

#### Task Checklist

- [x] Task 1.1: Build the DB Setup workflow (`db_setup/workflow.py`) using `MongoDBNode` with `create_index` to create `registered_users` (index on `external_userid`) and `user_records` (compound index on `{external_userid, record_date}`)
  - Dependencies: MongoDB instance available
  - Note: `create_index` implicitly creates the database and collection if they don't exist; operation is idempotent
- [x] Task 1.2: Create `db_setup/workflow_config.json` with `reminder_database` and collection name configurables
  - Dependencies: Task 1.1
- [x] Task 1.3a: Upload and run DB Setup workflow with `orcheo workflow upload` and `orcheo workflow run` to provision MongoDB
  - Dependencies: Task 1.2, Task 1.4 (mdb_connection_string in vault)
- [x] Task 1.3: Configure WeCom Customer Service account and obtain `corp_id`, `app_secret`, `token`, `encoding_aes_key`
  - Dependencies: WeCom admin access
- [x] Task 1.4: Store all secrets in Orcheo vault using `orcheo credential create`
  - Dependencies: Task 1.3
  - Credentials: `wecom_app_secret_medical_reminder`, `wecom_token`, `wecom_encoding_aes_key`, `mdb_connection_string`, `openai_api_key`
- [x] Task 1.5: Prepare workflow config JSON with `corp_id`, `reminder_database`, collection names
  - Dependencies: Task 1.3, Task 1.1

---

### Milestone 2: Message Handler Workflow

**Description:** Implement the webhook-triggered workflow that handles user registration, deregistration, and status reporting via an AI agent. Success criteria: all four user interaction flows pass integration tests.

#### Task Checklist

- [x] Task 2.1: Scaffold the Message Handler workflow file using `orcheo code template`
  - Dependencies: Milestone 1
- [x] Task 2.2: Implement WeCom ingress nodes: WebhookTriggerNode, WeComEventsParserNode, WeComAccessTokenNode, WeComCustomerServiceSyncNode
  - Dependencies: Task 2.1
- [x] Task 2.3: Implement the `ExtractAgentReplyNode` custom TaskNode
  - Dependencies: Task 2.1
- [x] Task 2.4: Configure the AgentNode with system prompt, `mongodb_find` and `mongodb_update_one` predefined tools
  - Dependencies: Task 2.2, Task 2.3
- [x] Task 2.5: Implement conditional routing edges (immediate response, CS sync should_process)
  - Dependencies: Task 2.2
- [x] Task 2.6: Wire the reply path: ExtractAgentReplyNode -> WeComCustomerServiceSendNode -> END
  - Dependencies: Task 2.3, Task 2.4
- [x] Task 2.7: Upload workflow with `orcheo workflow upload` and test registration flow
  - Dependencies: Task 2.6
- [x] Task 2.8: Test deregistration flow (with confirmation dialogue across two invocations)
  - Dependencies: Task 2.7
- [x] Task 2.9: Test status report flow (verify `user_records` document creation)
  - Dependencies: Task 2.7
- [x] Task 2.10: Test edge cases: duplicate registration, unregistered user reporting, ambiguous messages
  - Dependencies: Task 2.7

---

### Milestone 3: Daily Reminder Workflow

**Description:** Implement the cron-triggered workflow that sends personalised reminders to all active users daily at 9:00 AM Asia/Shanghai. Success criteria: reminders delivered to all active users in test runs.

#### Task Checklist

- [x] Task 3.1: Scaffold the Daily Reminder workflow file
  - Dependencies: Milestone 1
- [x] Task 3.2: Implement CronTriggerNode with `expression: "0 9 * * *"` and `timezone: "Asia/Shanghai"`
  - Dependencies: Task 3.1
- [x] Task 3.3: Implement WeComAccessTokenNode and MongoDBFindNode for fetching active users
  - Dependencies: Task 3.2
- [x] Task 3.4: Implement `PrepareIterationNode` custom TaskNode (initialise index and total count from find results)
  - Dependencies: Task 3.3
- [x] Task 3.5: Implement `SelectCurrentUserNode` custom TaskNode (pick user at current index, format personalised reminder message from their `reminder_items`)
  - Dependencies: Task 3.4
- [x] Task 3.6: Implement `IncrementCounterNode` custom TaskNode (advance index by 1)
  - Dependencies: Task 3.4
- [x] Task 3.7: Configure SubWorkflowNode containing WeComCustomerServiceSendNode for single-user message delivery
  - Dependencies: Task 3.5
- [x] Task 3.8: Wire the While edge loop: PrepareIteration -> [While: index < total] -> SelectCurrentUser -> SubWorkflow -> IncrementCounter -> [loop back]
  - Dependencies: Task 3.5, Task 3.6, Task 3.7
- [x] Task 3.9: Upload workflow and test with mock data (0 users, 1 user, multiple users)
  - Dependencies: Task 3.8
- [x] Task 3.10: Schedule the workflow with `orcheo workflow schedule` and verify cron execution
  - Dependencies: Task 3.9

---

### Milestone 4: Integration Testing & Deployment

**Description:** End-to-end testing with real WeCom Customer Service, verify both workflows work together, and deploy to production. Success criteria: full user lifecycle tested and workflows running in production.

#### Task Checklist

- [x] Task 4.1: Configure WeCom Customer Service webhook URL pointing to the Message Handler workflow
  - Dependencies: Milestone 2
- [x] Task 4.2: End-to-end test: register a user via WeChat -> verify MongoDB entry -> receive 9 AM reminder -> submit status report -> verify record
  - Dependencies: Milestone 2, Milestone 3, Task 4.1
- [x] Task 4.3: End-to-end test: deregister a user -> verify status set to inactive -> verify no reminder on next cron run
  - Dependencies: Task 4.2
- [x] Task 4.4: Verify cron schedule is active for the Daily Reminder workflow
  - Dependencies: Task 4.2, Task 4.3
- [ ] Task 4.5: Monitor first week of production usage and address any issues
  - Dependencies: Task 4.4

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-23 | ShaojieJiang | Initial draft |
| 2026-02-23 | ShaojieJiang | Replaced manual DB setup tasks (1.1, 1.2) with DB Setup workflow tasks |
