from jiwer import wer, cer, process_words
import os

def evaluate_predictions(reference_dir, hypothesis_dir):
    references = []
    hypotheses = []
    
    # فرض می‌کنیم نام فایل‌ها در هر دو پوشه یکی است
    ref_files = [f for f in os.listdir(reference_dir) if f.endswith('.txt')]
    
    for file in ref_files:
        with open(os.path.join(reference_dir, file), 'r') as f:
            references.append(f.read().strip())
        with open(os.path.join(hypothesis_dir, file), 'r') as f:
            hypotheses.append(f.read().strip())
            
    # محاسبه متریک‌های کلی
    total_wer = wer(references, hypotheses)
    total_cer = cer(references, hypotheses)
    
    print("=== Benchmark Results ===")
    print(f"Overall WER: {total_wer:.2%}")
    print(f"Overall CER: {total_cer:.2%}\n")
    
    print("=== Error Analysis (S/D/I) ===")
    for ref, hyp in zip(references[:5]): # بررسی ۵ خطای اول به عنوان نمونه
        output = process_words(ref, hyp)
        if output.wer > 0:
            print(f"REF: {ref}")
            print(f"HYP: {hyp}")
            print(f"Substitutions (S): {output.substitutions}, Deletions (D): {output.deletions}, Insertions (I): {output.insertions}")
            print("-" * 30)

if __name__ == "__main__":
    # آدرس پوشه متون طلایی (Ground truth) و متون تولید شده توسط مدل خودت را بده
    evaluate_predictions("path_to_ground_truth", "path_to_transcripts")