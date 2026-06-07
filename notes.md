## AI Usage


### Overall design

My overall design was quite simple: I added basic exploration tools for the agent, and then basic knowledge base information. I based everything on a mix of a) my use of claude code, and b) my own personal use of databricks.

I basically framed my work based into the following two steps.

### Step 1: Basic tools

I started by adding 3 tool:, explore schema (list schemas in duckdb), explore table (explore a particular table) and run_query (run a query without submitting it as an answer). 

In this step, I saw two major errors appear.

First, I would frequently see gpt-oss generate invalid sql or function calls like so: 
```json
[
  {
    "id": "chatcmpl-tool-966fe2d8785f205c",
    "type": "function",
    "function": {
      "name": "explore_table",
      "arguments": "{\n  \"table_name\": \"financial.account\",\n  \"limit\": 5\n"   ← no closing brace
    }
  },
  {
    "id": "",
    "type": "function",
    "function": {
      "name": "",            ← phantom: empty name
      "arguments": "}"        ← the missing brace leaked into a second call
    }
  }
]
```
To deal with improper sql, I added a json post-processing function. In step 2, I ended up using pydantic which was a lot cleaner

Second, I regularly saw my model exceed it's "max iteration" budget of 30. It was overly focused on exploring. In response, I did the following. A)  To deal with budget exceeding, I prepended the {current_iter} / {max_iter} to each model response, and explicitly wrote how many iterations the model had left. B) I also ammended the system prompt to remind the model that it should focus more on exploring early in its iteration cycle and submission later.

I saw approximately the final results at the end of step 1. In particular, I was quite shocked because I hadn't added a knowledge base.

```markdown
For reference, current standing:
- Easy: ~73% (47/64), bounded by SQL correctness variance.
- Hard: ~30% (19/64), bounded by missing access to the business-rule guides.
``` 

### Step 2: Adding a basic knowledge base

Next, I added the knowledge base compoennt. Interestingly, when I added these, I actually saw performance drop quite a bit. I initially went from a 70% -> a 38% pass rate on the easy problems, before later recovering.

- First iteration: Basic knowledge base
  - tools: explore_guides (list all the guides, with a 'grep' parameter), open_guide (let the agent read a guide)
- Second iteration: Tweaking iteration
  - I started using pydantic's partial json parsing (https://pydantic.dev/docs/validation/latest/concepts/json/#partial-json-parsing) rather than manually parsing everything which actually substantially boosted performance a lot
  - I updated the system prompt, and give it a really really short context
  - tools: list_guides, open_guide (removed the "after" parameter)
    - I renamed from explore_guides -> list_guides because it felt more natrual to me

```
┌───────┬────────┬──────────┬───────┬───────┬───────────┐
│ Split │ Passed │ Mismatch │ Other │ Total │ Pass Rate │
├───────┼────────┼──────────┼───────┼───────┼───────────┤
│ easy  │ 49     │ 12       │ 3     │ 64    │ 76.6%     │
├───────┼────────┼──────────┼───────┼───────┼───────────┤
│ hard  │ 25     │ 36       │ 3     │ 64    │ 39.1%     │
├───────┼────────┼──────────┼───────┼───────┼───────────┤
│ Total │ 74     │ 48       │ 6     │ 128   │ 57.8%     │
└───────┴────────┴──────────┴───────┴───────┴───────────┘
```

Andrew (my recruiter) told me to aim for 70% pass rate on easies, and 30% pass rate on the hards. I figured since I was beating that, and was at roughly the ~3 hour marker that I should call it quits.

### What I would do if I had more time

Three things come to mind:

First, I would experiment with using bash commands rather than standard sql . Anecdotally, agents are really good at using bash (https://www.reddit.com/r/AI_Agents/comments/1i2olbq/using_bash_scripting_to_get_ai_agents_make/).

Second, gpt-oss is a solid model but there are better options out there. I'd also explore iterating on the model itself. But for now, I didn't want to waste money on the prompting.

Third, I would also iterate a bit on the knowledge base. I saw a small boost. But not a significant one. We were still seeing AGENT_ERROR, and one SQL_ERROR. So taking a look at what we could do to tweak agent reliability in that case would go a long way.

## How I used AI?

I used Claude code to write all of my code. 

I usually include something in my prompt to tell claude to "grill" me: https://www.aihero.dev/my-grill-me-skill-has-gone-viral. So it's extremely aggressive in the questioning

I probably spent about 4 hours on this assignment, and I probably used $30-40 of claude-opus tokens? I spent another 30 minutes cleaning up my writeup.
