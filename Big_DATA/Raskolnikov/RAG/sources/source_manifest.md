# Source manifest

- `cp_tr_pdf_main` | primary_or_translation_pdf | tr | local_workspace_pdf | parse_into_rag | `RATA/Fyodor Mihailoviç Dostoyevski - Suç ve Ceza.pdf`
- `cp_tr_pdf_alt` | primary_or_translation_pdf | tr | local_workspace_pdf | parse_into_rag | `RATA/Fyodor Mihailoviç Dostoyevski - Suç ve Ceza (1).pdf`
- `cp_tr_pdf_scan` | primary_or_translation_pdf | tr | local_workspace_pdf | parse_into_rag | `RATA/20114324_suc-ve-ceza-dostoyovski.pdf`
- `raskolnikov_criticism_nietzsche` | character_criticism_pdf | tr | local_workspace_pdf | parse_into_rag | `RATA/Raskolnikov’un Ahlaki İkilemi Nietzscheci Bir Bakış[#647919]-950756.pdf`
- `raskolnikov_legal_reading` | character_criticism_pdf | tr | local_workspace_pdf | parse_into_rag | `RATA/SuveCezaRomanndaRaskolnikovunlemiOlduuFiillerinTrkCezaKanunundaDzenlenenSuTipleriAsndanDeerlendirilmesi.pdf`
- `legacy_rata_train` | legacy_lora_dataset | mixed | existing_repo_asset | audit_selectively | `data/RATA_dataset/train.jsonl`
- `legacy_rata_valid` | legacy_lora_dataset | mixed | existing_repo_asset | audit_selectively | `data/RATA_dataset/valid.jsonl`
- `legacy_hq_train` | legacy_lora_dataset | mixed | existing_repo_asset | audit_selectively | `data/Raskolnikov_HQ_Dataset/train.jsonl`
- `legacy_hq_valid` | legacy_lora_dataset | mixed | existing_repo_asset | audit_selectively | `data/Raskolnikov_HQ_Dataset/valid.jsonl`
- `legacy_prompts` | legacy_prompt_framing | en | existing_repo_asset | reference_only | `prompts.json`
- `legacy_augment_script` | legacy_prompt_framing | en | existing_repo_asset | reference_only | `augment_dataset.py`
- `legacy_app_prompt` | legacy_prompt_framing | en | existing_repo_asset | reference_only | `main.py`
