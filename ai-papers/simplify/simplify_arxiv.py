import torch
import argparse
import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple
from threading import Thread
from transformers import TextIteratorStreamer, MllamaForCausalLM, AutoTokenizer
import sys

class BlogPostGenerator:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.model = MllamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Model is running on: {self.model.device}")

    def _parse_frontmatter(self, content: str) -> Tuple[Dict, str]:
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)
        if not frontmatter_match:
            return {}, content
        try:
            metadata = yaml.safe_load(frontmatter_match.group(1))
            return metadata, frontmatter_match.group(2)
        except yaml.YAMLError:
            return {}, content

    def _extract_paper_summaries(self, content: str) -> List[Dict]:
        """
        Extracts paper summaries from content that follows the format:
        #### Title
        *Main Problem:* ...
        *Approach:* ...
        ...
        ArXiv: [id](link)
        """
        # First, get everything after "# Summaries"
        summaries_section = re.search(r'(?s)# Summaries\s*(.*?)(?=\Z|\n---)', content)
        if not summaries_section:
            return []
            
        content = summaries_section.group(1)
        
        # Pattern to match each paper section
        paper_pattern = r'#### (.*?)\n\n(.*?)(?=\n\nArXiv: \[(.*?)\]\((.*?)\))'
        
        # Pattern to extract upvotes from title
        upvotes_pattern = r'↑(\d+)$'
        
        summaries = []
        matches = re.finditer(paper_pattern, content, re.DOTALL)
        
        for match in matches:
            title = match.group(1).strip()
            content = match.group(2).strip()
            arxiv_id = match.group(3).strip()
            arxiv_link = match.group(4).strip()
            
            # Extract upvotes from title
            upvotes_match = re.search(upvotes_pattern, title)
            upvotes = int(upvotes_match.group(1)) if upvotes_match else 0
            
            # Clean title by removing upvote count
            clean_title = re.sub(r'\s*↑\d+$', '', title)
            
            summaries.append({
                'title': clean_title,
                'content': content,
                'arxiv_id': arxiv_id,
                'arxiv_link': arxiv_link,
                'upvotes': upvotes
            })
        
        # Sort by upvotes in descending order
        summaries.sort(key=lambda x: x['upvotes'], reverse=True)
        
        return summaries

    def generate_text_stream(self, prompt: str, max_new_tokens: int = 4096) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        try:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.7,
                temperature=0.2,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                print(new_text, end='', flush=True)

            thread.join()
            
            return generated_text.strip()
        finally:
            del inputs
            torch.cuda.empty_cache()

    def summarize_papers(self, papers: List[Dict]) -> str:
        # Extract summaries section from content using regex
        summaries_pattern = r'#### (.*?)\n\n(.*?)(?=(?:\n\n#### |\Z))'
        
        papers_text = []
        for paper in papers:
            content = paper['content']
            
            # Extract key components using regex
            main_problem = re.search(r'\*Main Problem:\* (.*?)\n', content)
            approach = re.search(r'\*Approach:\* (.*?)\n', content)
            findings = re.search(r'\*Findings:\* (.*?)\n', content)
            key_results = re.search(r'\*Key Results:\*(.*?)(?:\*|$)', content, re.DOTALL)
            
            formatted_content = f"""#### {paper['title']}

                *Main Problem:* {main_problem.group(1) if main_problem else 'Not specified'}

                *Approach:* {approach.group(1) if approach else 'Not specified'}

                *Findings:* {findings.group(1) if findings else 'Not specified'}

                *Key Results:*
                {key_results.group(1).strip() if key_results else 'Not specified'}

                ArXiv: [{paper['arxiv_id']}]({paper['arxiv_link']})
                """
            papers_text.append(formatted_content)

            all_papers = "\n\n".join(papers_text)
        
        prompt = f"""# AI Papers of the Week

            Task: Create a condensed version of the summaries below. For each paper:
            1. Keep the header format (####)
            2. Extract and summarize:
            - The main problem being addressed
            - Key findings and quantitative results that are explicitly stated
            - Notable metrics and measurements
            3. Preserve the ArXiv link
            4. Maintain the original paper order
            5. Use markdown formatting

            DO NOT:
            - Add any information not present in the original summaries
            - Add introductory or concluding text
            - Make up new statistics or findings
            - Add commentary or analysis

            Here are the papers to summarize:

            {all_papers}"""
        
        return self.generate_text_stream(prompt, max_new_tokens=4096)

    def generate_blog_post(self, input_path: Path) -> str:
        content = input_path.read_text()
        metadata, content = self._parse_frontmatter(content)
        
        # Clean up the content
        content = re.sub(r'<iframe.*?</iframe>\s*', '', content, flags=re.DOTALL)
        
        papers = self._extract_paper_summaries(content)
        print(papers)
        
        return self.summarize_papers(papers)

def main():
    parser = argparse.ArgumentParser(description="Generate a simplified blog post from QMD file using Llama")
    parser.add_argument("input_file", type=str, help="Input QMD file path")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct", 
                        help="Llama model to use")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return

    try:
        generator = BlogPostGenerator(model_id=args.model)
        blog_content = generator.generate_blog_post(input_path)
        
        # Save the output
        output_path = input_path.parent / 'blog_post.qmd'
        output_path.write_text(blog_content)
        print(f"\nBlog post generated and saved to: {output_path}")
    except Exception as e:
        print(f"Error generating blog post: {str(e)}")

if __name__ == "__main__":
    main()