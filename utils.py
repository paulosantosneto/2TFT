import json

def export_strategies_to_jsonl(results, questions_text):

    all_generated_files = []


    def export_mode(mode):

        data_greedy = []
        data_cot_max = []
        data_cot_sum = []
        data_self_consistency = []

        for idx, item in enumerate(results):
            question = questions_text[idx] if idx < len(questions_text) else ""
            candidates = item['candidates']

            ans_key = f"{mode}_answer"
            score_key = f"{mode}_score"

            valid_candidates = [
                c for c in candidates
                if c[ans_key] != "[invalid]" and c[ans_key] is not None
            ]

            if not valid_candidates:
                fallback = candidates[0]
                common_entry = {
                    "question": question,
                    "answer": fallback[ans_key],
                    "reasoning": fallback['text'],
                    "score": fallback[score_key]
                }

                data_greedy.append(common_entry)
                data_cot_max.append(common_entry)
                data_cot_sum.append(common_entry)
                data_self_consistency.append(common_entry)
                continue

            greedy_cand = candidates[0]
            data_greedy.append({
                "question": question,
                "answer": greedy_cand[ans_key],
                "reasoning": greedy_cand['text']
            })

            best_max = max(valid_candidates, key=lambda x: x[score_key])
            data_cot_max.append({
                "question": question,
                "answer": best_max[ans_key],
                "reasoning": best_max['text'],
                "score": best_max[score_key]
            })

            score_sum = {}
            representative_text = {}

            for cand in valid_candidates:
                ans = cand[ans_key]
                score_sum[ans] = score_sum.get(ans, 0) + cand[score_key]

                if ans not in representative_text or cand[score_key] > representative_text[ans][score_key]:
                    representative_text[ans] = cand

            best_sum_ans = max(score_sum, key=score_sum.get)

            data_cot_sum.append({
                "question": question,
                "answer": best_sum_ans,
                "reasoning": representative_text[best_sum_ans]['text'],
                "aggregated_score": score_sum[best_sum_ans]
            })

            counts = collections.Counter([c[ans_key] for c in valid_candidates])
            best_sc_ans = counts.most_common(1)[0][0]

            sc_text = next(
                c['text'] for c in valid_candidates
                if c[ans_key] == best_sc_ans
            )

            data_self_consistency.append({
                "question": question,
                "answer": best_sc_ans,
                "reasoning": sc_text,
                "votes": counts[best_sc_ans]
            })


        def save_jsonl(filename, data):
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"File saved: {filename} ({len(data)} records)")
            all_generated_files.append(filename)


        save_jsonl(f"dataset_greedy_{mode}.jsonl", data_greedy)
        save_jsonl(f"dataset_cot_max_{mode}.jsonl", data_cot_max)
        save_jsonl(f"dataset_cot_sum_{mode}.jsonl", data_cot_sum)
        save_jsonl(f"dataset_self_consistency_{mode}.jsonl", data_self_consistency)

    export_mode("regex")
    export_mode("bert")

    final_zip = "all_datasets.zip"

    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in all_generated_files:
            z.write(file)

    print(f"\nFINAL ZIP created: {final_zip}")

    