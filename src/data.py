from datasets import load_dataset

def load_pubmedqa():
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    questions = [row['question'] for row in ds]
    abstract = [' '.join(row['context']['contexts']) for row in ds]
    return questions, abstract

def load_labels():
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split= 'train')
    return [row['final_decision'] for row in ds]