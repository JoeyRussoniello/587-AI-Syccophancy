## Code Quality (Chores)

- [X] Write initial text generation scaffolding (Brian)
- [ ] Refactor Generate Responses for dependency injection / modify in one place (Joey)
  - [ ] Generate Prompt Protocol
  - [ ] Overall function that uses this prtoocl
- [ ] Migrate from CSV to SQLite Relational DB **Complete This Today** (Joey)
  - [ ] LLM_Responses (id, prompt_id, system_prompt_id, model, response)
  - [ ] SystemPrompts (system_prompt_id, system_prompt)
  - [ ] Prompts (prompt_id, prompt, YTA_NTA, Flipped, Validation)

## Gather LLM Data:

### Base (System prompt. No stress on anti-sycophancy) **Gather By 4/10 (Fri)** 

- [ ] Claude (Brian)
- [ ] OpenAI (Joey)
- [ ] Gemini (Joey)

### Resistant System Prompt **Gather by 4/13 (Mon)**

- [ ] Claude (Brian)
- [ ] OpenAI (Joey)
- [ ] Gemini (Joey)
  
### Resistant System Prompt + Few Shot Examples **Gather by 4/13 (Mon)**

## Exploratory Data Analytics 

- [ ] Examine Rates of false NTA (sycophantic) responses for each model - 4/15 Wed (Brian) 
- [ ] Use SBERT to encode prompts (maybe cache in DB), analyze prompts for similarity - 4/20 Mon (Brian & Joey)
- [ ] Use BertScore to compare LLM Reasoning to Crowdsourced Reasoning  - 4/20 Mon (Brian & Joey)
- [ ] Qualitative Analysis. Do we see any similarity on when the AI fails to point out bad behavior. Pick examples - 4/22 (Wed) (Brian & Joey)

## Additional Steps (If Time Allows):

Convert Converation to Multi-step (split by periods)

## Formalization Steps

- Presentation (Mon April 27th)
  - [ ] So what? Who cares (from annotated bibs)
  - [ ] So what? Who cares (from annotated bibs)
  - [ ] Our Methods 
  - [ ] Results (graphs, numbers, etc)
  - [ ] Discussion (impacts of system prompting, what this means for future research)
  - [ ] Challenges 
  - [ ] Conclusion
- Paper (Thursday April 30th)
  - [ ] Introduction (Joey)
  - [ ] Methodology (Joey)
  - [ ] Results (Brian)
  - [ ] Discussion (Brian)
  - [ ] Limitations / Future Resarch (Brian)
  - [ ] Conclusion (Joey)