# Requirements Document

## METADATA
- **Authors:** ShaojieJiang
- **Project/Feature Name:** WeChat Medical Reminder
- **Type:** Product
- **Summary:** An Orcheo-powered WeChat chatbot that manages user registration for personalised medical reminders, sends daily scheduled reminders, and collects/stores user status reports via WeCom Customer Service.
- **Owner:** ShaojieJiang
- **Date Started:** 2026-02-23

## RELEVANT LINKS & STAKEHOLDERS

| Documents | Link | Owner | Name |
|-----------|------|-------|------|
| Orcheo Platform Docs | Internal | Engineering | ShaojieJiang |
| WeCom Customer Service API | WeChat Work docs | Engineering | ShaojieJiang |

## PROBLEM DEFINITION

### Objectives
Build an automated WeChat-based medical reminder system that allows users to register personalised health-check items, receive daily reminders at 9:00 AM (Asia/Shanghai), and report their status through natural-language conversation handled by an AI agent.

### Target users
Individuals who need daily medical reminders (e.g., medication intake, blood pressure measurement, exercise routines) and prefer interacting through WeChat.

### User Stories

| As a... | I want to... | So that... | Priority | Acceptance Criteria |
|---------|--------------|------------|----------|---------------------|
| WeChat user | Send a message with my reminder items and phone number to register | I start receiving daily personalised reminders | P0 | Agent creates entry in `registered_users` if the user does not already exist; confirms registration to user |
| Registered user | Send a message expressing my wish to deregister | I stop receiving reminders and my data is removed | P0 | Agent confirms intent before deregistering; sets user status to inactive in `registered_users`; confirms completion |
| Registered user | Receive a daily reminder at 9:00 AM with my personalised items | I am prompted to complete my health tasks and report back | P0 | All active registered users receive a personalised reminder message every day at 09:00 Asia/Shanghai |
| Registered user | Reply to the reminder with my status in free-form text | My daily health record is saved for future reference | P0 | Agent parses the reply and stores a structured record in `user_records`; confirms receipt |
| Admin | Run a setup workflow to create the required MongoDB database and collections with proper indexes | The infrastructure is ready before deploying the main workflows | P0 | Running the DB Setup workflow creates `registered_users` (index on `external_userid`) and `user_records` (compound index on `{external_userid, record_date}`) collections; workflow is idempotent |
| Registered user | Update my reminder items or phone number | My registration reflects the latest preferences | P1 | Agent updates the existing `registered_users` entry |

### Context, Problems, Opportunities
Many people need daily health reminders (medication, measurements, exercises) but forget without prompting. Existing reminder apps require installing separate software, whereas WeChat is already used daily by the target audience. By building on WeCom Customer Service (微信客服), external WeChat users can interact directly without needing a WeCom account, lowering adoption friction.

### Product goals and Non-goals

**Goals:**
- Provide frictionless registration via natural-language WeChat messages
- Deliver reliable daily reminders at a fixed time with personalised content
- Collect and persist daily health status reports for each user
- Leverage an AI agent (LLM) for natural-language understanding to handle registration, deregistration, and status reporting without rigid command syntax

**Non-goals:**
- Medical diagnosis or health advice (the system is a reminder/tracker only)
- Multi-language support (Chinese-only for WeChat audience)
- Rich media messages (text-only for MVP)
- Analytics dashboard or reporting UI (out of scope for initial release)
- Integration with external health platforms or EHR systems

## PRODUCT DEFINITION

### Requirements

#### P0 - MVP

1. **User Registration Flow**
   - Accept free-form text messages containing reminder items and phone number
   - Agent extracts structured data: reminder items list and phone number
   - Check if user already exists in `registered_users` (by `external_userid`)
   - If new user: insert into MongoDB with status `active`
   - If existing active user: inform user they are already registered
   - Confirm registration with a summary of stored items

2. **User Deregistration Flow**
   - Detect deregistration intent from user message
   - Agent asks for explicit confirmation before proceeding
   - On confirmation: set user status to `inactive` in `registered_users`
   - Confirm deregistration to user

3. **Daily Reminder Schedule**
   - Cron trigger at `0 9 * * *` in `Asia/Shanghai` timezone
   - Fetch all users with `status: "active"` from `registered_users`
   - For each user, send a personalised WeChat message listing their reminder items and asking them to report status
   - Use SubWorkflow per user for message delivery

4. **User Status Report Collection**
   - Accept free-form text replies from users as status reports
   - Agent processes the reply and extracts relevant status information
   - Store each report as a new document in `user_records` with the user's ID, date, and processed data
   - Confirm receipt to user

5. **Database & Collection Setup (Admin Workflow)**
   - A dedicated Orcheo workflow that admins run once (or re-run idempotently) to provision the MongoDB infrastructure
   - Creates the `registered_users` collection with an index on `external_userid`
   - Creates the `user_records` collection with a compound index on `{external_userid, record_date}`
   - Uses Orcheo's built-in `MongoDBNode` with `create_index` operation (which implicitly creates database and collection if they don't exist)
   - Configurable database and collection names via `workflow_config.json`

#### P1 - Future Enhancements

5. **Update Registration Details**
   - Allow users to modify their reminder items or phone number via conversation
   - Agent updates the existing `registered_users` document

6. **Missed Reminder Follow-up**
   - If a user hasn't reported by a configurable time, send a follow-up reminder

### Designs
Not applicable - text-based chat interface only, no custom UI.

## TECHNICAL CONSIDERATIONS

### Architecture Overview

The system consists of two Orcheo workflows sharing two MongoDB collections:

1. **Message Handler Workflow** (webhook-triggered): Processes all incoming WeChat messages through WeCom Customer Service. Uses an AgentNode with MongoDB tools to handle registration, deregistration, and status report collection.

2. **Daily Reminder Workflow** (cron-triggered): Runs daily at 9:00 AM Asia/Shanghai. Fetches all active registered users from MongoDB and sends personalised reminders via SubWorkflow per user.

Both workflows connect to external WeChat users through WeCom Customer Service (微信客服) and share:
- `registered_users` MongoDB collection
- `user_records` MongoDB collection

### Technical Requirements

- **Orcheo Platform**: Workflow engine with webhook and cron triggers
- **WeCom Customer Service**: For bidirectional messaging with external WeChat users
- **MongoDB**: Two collections for persistent storage
- **LLM (OpenAI GPT-4o-mini or equivalent)**: For natural-language understanding in the agent
- **Orcheo Vault Secrets**: WeCom credentials, MongoDB connection string, OpenAI API key

### AI/ML Considerations

#### Data Requirements
No training data required. The system uses a pre-trained LLM (e.g., GPT-4o-mini) with prompt engineering to handle:
- Intent classification (register / deregister / report status / other)
- Entity extraction (reminder items, phone number, status data)
- Confirmation dialogue management (deregistration flow)

#### Algorithm selection
Pre-trained LLM via AgentNode with tool-calling capabilities. The agent uses `mongodb_find` and `mongodb_update_one` predefined tools to read and write MongoDB directly, guided by a detailed system prompt.

#### Model performance requirements
- Intent recognition accuracy: > 95% for the four main intents
- Entity extraction: reliable extraction of phone numbers and reminder item lists from free-form Chinese text
- Latency: agent response within 10 seconds for interactive flows

## LAUNCH/ROLLOUT PLAN

### Success metrics

| KPIs | Target & Rationale |
|------|--------------------|
| [Primary] Daily active users reporting status | > 80% of registered users report daily |
| [Secondary] Registration completion rate | > 95% of registration attempts succeed |
| [Guardrail] False deregistration rate | < 1% of deregistrations are unintended |

### Rollout Strategy
1. Deploy both workflows to Orcheo local instance for testing
2. Configure WeCom Customer Service account and webhook
3. Internal team testing with a small group of users
4. Gradual rollout to target user group

## HYPOTHESIS & RISKS

**Hypothesis:** Users who receive personalised daily reminders via their existing WeChat app will achieve higher adherence to medical routines compared to generic reminder apps, because the interaction is embedded in their daily communication tool and requires no additional app installation.

**Risk 1: WeCom API rate limits.** Sending reminders to many users simultaneously may hit WeCom API rate limits.
- *Mitigation:* Implement backpressure in the SubWorkflow iteration with configurable delays between sends.

**Risk 2: LLM misinterpretation.** The agent may misclassify intent or extract incorrect data from ambiguous messages.
- *Mitigation:* Require explicit confirmation for destructive actions (deregistration). Include validation checks before MongoDB writes.

**Risk 3: User sends status report outside reminder window.** Users may report at arbitrary times, making it unclear which day's reminder they are responding to.
- *Mitigation:* Default to current date for record association. The agent can ask for clarification if the message is ambiguous.

## APPENDIX
