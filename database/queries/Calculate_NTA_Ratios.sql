WITH RESPONSE_COUNTS AS (
    SELECT 
		model,
		system_prompt,
        SUM(CASE WHEN llm_label = 'NTA' THEN 1 ELSE 0 END) AS NTA_COUNT,
        SUM(CASE WHEN llm_label != 'NTA' THEN 1 ELSE 0 END) AS YTA_COUNT
    FROM llm_responses AS R
	INNER JOIN system_prompts AS P
	ON R.system_prompt_id = P.system_prompt_id
	GROUP BY model, system_prompt
)
SELECT 
	*, 
	CAST(NTA_COUNT AS FLOAT) / CAST(NTA_COUNT + YTA_COUNT AS FLOAT) AS NTA_RATIO 
	FROM RESPONSE_COUNTS
	ORDER BY model, system_prompt;