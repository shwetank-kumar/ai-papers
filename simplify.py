import yaml
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import TextIteratorStreamer, MllamaForCausalLM, AutoTokenizer
import torch

def init_model(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    model = MllamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Model is running on: {model.device}")
    return model, tokenizer

def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not frontmatter_match:
        return {}, content
    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
        return metadata, frontmatter_match.group(2)
    except yaml.YAMLError:
        return {}, content

def extract_paper_summaries(content: str) -> List[Dict]:
    summaries_section = re.search(r'(?s)# Summaries\s*(.*?)(?=\Z)', content)
    if not summaries_section:
        print("No summaries section found")
        return []
        
    content = summaries_section.group(1)
    paper_pattern = r'#### (.*?)\n\n(.*?)(?=\n\nArXiv: \[(.*?)\]\((.*?)\))'
    upvotes_pattern = r'↑(\d+)$'
    
    summaries = []
    matches = re.finditer(paper_pattern, content, re.DOTALL)
    
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        arxiv_id = match.group(3).strip()
        arxiv_link = match.group(4).strip()
        
        upvotes_match = re.search(upvotes_pattern, title)
        upvotes = int(upvotes_match.group(1)) if upvotes_match else 0
        clean_title = re.sub(r'\s*↑\d+$', '', title)
        
        summaries.append({
            'title': clean_title,
            'content': content,
            'arxiv_id': arxiv_id,
            'arxiv_link': arxiv_link,
            'upvotes': upvotes
        })
    
    summaries.sort(key=lambda x: x['upvotes'], reverse=True)
    return summaries

import yaml
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not frontmatter_match:
        return {}, content
    try:
        metadata = yaml.safe_load(frontmatter_match.group(1))
        return metadata, frontmatter_match.group(2)
    except yaml.YAMLError:
        return {}, content

def extract_paper_summaries(content: str) -> List[Dict]:
    summaries_section = re.search(r'(?s)# Summaries\s*(.*?)(?=\Z)', content)
    if not summaries_section:
        print("No summaries section found")
        return []
        
    content = summaries_section.group(1)
    paper_pattern = r'#### (.*?)\n\n(.*?)(?=\n\nArXiv: \[(.*?)\]\((.*?)\))'
    upvotes_pattern = r'↑(\d+)$'
    
    summaries = []
    matches = re.finditer(paper_pattern, content, re.DOTALL)
    
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        arxiv_id = match.group(3).strip()
        arxiv_link = match.group(4).strip()
        
        upvotes_match = re.search(upvotes_pattern, title)
        upvotes = int(upvotes_match.group(1)) if upvotes_match else 0
        clean_title = re.sub(r'\s*↑\d+$', '', title)
        
        # Parse sections from content
        sections = {}
        current_section = None
        lines = []
        
        for line in content.split('\n'):
            if line.startswith('*') and ':' in line:
                if current_section and lines:
                    sections[current_section] = '\n'.join(lines).strip()
                current_section = line.split(':')[0].strip('*').strip()
                lines = [line.split(':', 1)[1].strip()]
            else:
                lines.append(line.strip())
                
        if current_section and lines:
            sections[current_section] = '\n'.join(lines).strip()
        
        summaries.append({
            'title': clean_title,
            'sections': sections,
            'arxiv_id': arxiv_id,
            'arxiv_link': arxiv_link,
            'upvotes': upvotes
        })
    
    summaries.sort(key=lambda x: x['upvotes'], reverse=True)
    return summaries

def generate_summary(model, tokenizer, paper: dict) -> str:
    prompt = f"""Paper: {paper['title']}

Content:
{paper['sections']}

Write a single witty paragraph that explains:
1. The key problem and solution
2. The exact quantitative results as stated in the paper
3. Why this matters for real applications

Use exact numbers from the paper. Be entertaining but factual. Write ~500 words max."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up any meta-commentary
        summary = response.replace(prompt, "").strip()
        summary = summary.split("Note:")[0].strip()  # Remove any notes
        return summary
    finally:
        del inputs
        torch.cuda.empty_cache()
        
def analyze_papers(papers: list, model, tokenizer, input_filename: str, frontmatter: dict, original_content: str):
    output_filename = input_filename.rsplit('.', 1)[0] + '_simplified.' + input_filename.rsplit('.', 1)[1]
    
    # Get everything before "# Summaries"
    pre_summaries = original_content.split("# Summaries")[0]
    
    summaries = []
    for paper in papers:
        try:
            summary = generate_summary(model, tokenizer, paper)
            summaries.append({
                'title': paper['title'],
                'upvotes': paper['upvotes'],
                'summary': summary
            })
        except Exception as e:
            print(f"Error analyzing {paper['title']}: {str(e)}")
    
    with open(output_filename, 'w') as f:
        # Write exact content up to Summaries
        f.write(pre_summaries)
        
        # Write Summaries section
        f.write("# Summaries\n\n")
        for summary in summaries:
            f.write(f"## {summary['title']} (↑{summary['upvotes']})\n\n")
            f.write(f"{summary['summary']}\n\n")
            f.write("---\n\n")
    
    print(f"Full content written to: {output_filename}")
    
    
def main():
    parser = argparse.ArgumentParser(description="Parse and summarize papers")
    parser.add_argument("filename", help="Input file to parse")
    args = parser.parse_args()
    
    input_path = Path(args.filename)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return

    original_content = input_path.read_text()
    content = re.sub(r'<iframe.*?</iframe>\s*', '', original_content)
    
    metadata, content = parse_frontmatter(content)
    papers = extract_paper_summaries(content)
    model, tokenizer = init_model()
    
    analyze_papers(papers, model, tokenizer, args.filename, metadata, original_content)

if __name__ == "__main__":
    main()