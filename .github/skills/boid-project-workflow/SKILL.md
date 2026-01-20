---
name: boid-project-workflow
description: Use for all boid simulation project work - manages memory files for plans, queries, tests, and commits
---

# Boid Project Workflow

## Overview

This skill manages the boid simulation C++ project with persistent memory files. All work flows through these memory files to maintain project state and history.

**Announce at start:** "I'm using the boid-project-workflow skill."

## Memory Files

| File | Purpose |
|------|---------|
| `memory/plan.md` | Primary planning document - class designs, function prototypes, implementation steps |
| `memory/queries.md` | Log of all user queries/requests made to Copilot |
| `memory/tests.md` | Test specifications and test cases |
| `memory/commits.md` | Commit summaries after each commit |

## Workflow Steps

### 1. Log Query
**Every user request** must be logged to `memory/queries.md`:
```markdown
## [YYYY-MM-DD HH:MM] Query
<user request>
```

### 2. Update Plan
Before any implementation, update `memory/plan.md` with:
- Class definitions and prototypes
- Implementation steps
- Design decisions

### 3. Brainstorm & Design (use brainstorming skill)
For new features/classes:
- Define class structures
- Function prototypes with signatures
- Design rationale

### 4. Implement with TDD (use test-driven-development skill)
- Write tests first → document in `memory/tests.md`
- Implement minimal code
- Run and verify

### 5. Commit & Log
After each commit:
- Add summary to `memory/commits.md`:
```markdown
## [YYYY-MM-DD] <commit-hash-short>
**Message:** <commit message>
**Changes:** <brief description>
```

## Memory File Formats

### memory/plan.md
```markdown
---
name: plan
description: This is the file for the plan of the repository
---

## Current Phase
<phase description>

## Classes
### ClassName
- **Purpose:** <purpose>
- **Members:** <list>
- **Methods:** <prototypes>

## Implementation Steps
- [ ] Step 1
- [x] Step 2 (completed)
```

### memory/queries.md
```markdown
---
name: queries
description: file with prompts made to the copilot.
---

## [YYYY-MM-DD HH:MM] <brief title>
<full query text>
```

### memory/tests.md
```markdown
---
name: tests
description: Test specifications for the boid simulation
---

## TestSuiteName
### test_name
- **Input:** <input>
- **Expected:** <expected output>
- **Status:** PASS/FAIL/PENDING
```

### memory/commits.md
```markdown
---
name: commits
description: Commit history summaries
---

## [YYYY-MM-DD] <hash>
**Message:** <msg>
**Changes:** <summary>
```

## Remember
- Always log queries first
- Update plan before coding
- Document tests before implementation
- Log commits after successful commits
- Use TDD: red → green → refactor
