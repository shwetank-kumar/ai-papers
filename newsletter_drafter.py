import re
import yaml
import markdown
from pathlib import Path
from typing import Dict, List, Tuple

class NewsletterGenerator:
    def __init__(self):
        self.html_template = '''<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ 
                    font-family: 'Open Sans', Arial, sans-serif;
                    line-height: 1.6;
                    color: #2c3e50;
                    background-color: #f6f6f6;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 2rem;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 2rem;
                    background-color: #f6f6f6;
                    padding: 1.5rem;
                    border-radius: 8px;
                }}
                .header h1 {{
                    font-size: 2.5rem;
                    margin-bottom: 0.5rem;
                    color: #1a202c;
                }}
                .header h2 {{
                    font-size: 1.8rem;
                    color: #4a5568;
                    font-weight: 500;
                    margin-bottom: 1rem;
                }}
                .content {{ 
                    max-width: 100%;
                    font-size: 1.1rem;
                    line-height: 1.8;
                }}
                .content p {{
                    margin-bottom: 1.5rem;
                    color: #2d3748;
                }}
                .media-content {{
                    background: #ffffff;
                    padding: 2rem;
                    border-radius: 8px;
                    margin: 2rem 0;
                    text-align: center;
                }}
                .media-content h3 {{
                    font-size: 1.5rem;
                    color: #1a202c;
                    margin-bottom: 1rem;
                }}
                .media-content p {{
                    font-size: 1.1rem;
                    color: #4a5568;
                    margin-bottom: 1.5rem;
                }}
                .paper-summary {{
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                    padding: 2rem;
                    transition: box-shadow 0.3s ease;
                }}
                .paper-summary:hover {{ 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15); 
                }}
                .paper-title {{
                    font-size: 1.3rem;
                    line-height: 1.7;
                    color: #2d3748;
                    margin-bottom: 0.5rem;
                }}
                .paper-title strong {{
                    color: #1a202c;
                    font-size: 1.4rem;
                }}
                .paper-footer {{
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                    margin-top: 1.5rem;
                    padding-top: 1rem;
                    border-top: 1px solid #e2e8f0;
                }}
                .paper-metrics {{
                    margin-left: 1rem;
                    font-size: 1rem;
                    color: #4a5568;
                }}
                .newsletter-form {{
                    background-color: #ffffff;
                    border-radius: 8px;
                    padding: 2.5rem;
                    text-align: center;
                    margin: 2rem 0;
                }}
                .newsletter-form h4 {{
                    color: #1a202c;
                    font-size: 1.6rem;
                    margin-bottom: 1rem;
                }}
                .newsletter-form p {{
                    font-size: 1.1rem;
                    color: #4a5568;
                    margin-bottom: 1.5rem;
                }}
                .category {{
                    color: #4a5568;
                    background-color: #edf2f7;
                    padding: 4px 12px;
                    margin: 4px;
                    border-radius: 6px;
                    font-size: 0.9rem;
                    display: inline-block;
                    font-weight: 500;
                }}
                .read-more-section {{
                    background-color: #f8fafc;
                    border-radius: 8px;
                    padding: 2.5rem;
                    margin-top: 2.5rem;
                    text-align: center;
                }}
                .remaining-papers-header {{
                    font-size: 1.8rem;
                    font-weight: bold;
                    color: #2b6cb0;
                    margin-bottom: 1.5rem;
                    line-height: 1.3;
                }}
                .topics-list {{
                    list-style: none;
                    padding: 0;
                    margin: 1.5rem 0;
                    text-align: left;
                    display: inline-block;
                }}
                .topics-list li {{
                    margin: 0.75rem 0;
                    padding-left: 1.5rem;
                    position: relative;
                    font-size: 1.1rem;
                    color: #2d3748;
                }}
                .topics-list li:before {{
                    content: "•";
                    position: absolute;
                    left: 0;
                    color: #2b6cb0;
                }}
                .cta-button {{
                    display: block;
                    background-color: #1a202c !important;
                    color: #ffffff !important;
                    padding: 14px 28px;
                    border-radius: 6px;
                    text-decoration: none;
                    margin: 2rem auto 0;
                    font-weight: 600;
                    font-size: 1.1rem;
                    transition: all 0.3s ease;
                    width: fit-content;
                }}
                .cta-button:hover {{
                    background-color: #2d3748 !important;
                    transform: translateY(-2px);
                }}
                .paper-link {{
                    text-decoration: none;
                    color: #2b6cb0;
                    font-weight: 600;
                    padding: 8px 20px;
                    border-radius: 6px;
                    transition: all 0.3s ease;
                    display: inline-block;
                    line-height: normal;
                    height: fit-content;
                    font-size: 1rem;
                }}
                .paper-link:hover {{
                    background-color: #ebf4ff;
                    transform: translateX(5px);
                }}
                @media (max-width: 768px) {{
                    .container {{ padding: 1rem; }}
                    .paper-summary {{ padding: 1.5rem; }}
                    .header h1 {{ font-size: 2rem; }}
                    .header h2 {{ font-size: 1.5rem; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{emoji} {series_title}</h1>
                    <h2>{subtitle}</h2>
                    <div class="categories">
                        {categories}
                    </div>
                </div>
                <div class="content">
                    {introduction}
                    {paper_summaries}
                </div>
            </div>
        </body>
        </html>'''

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
        """Extract paper summaries from the QMD content"""
        summaries = []
        
        # Pattern to match the entire paper entry
        paper_pattern = r'\*\*(.*?)\*\*(.*?)\[arXiv:(.*?)\]\((.*?)\)\s*👍\s*(\d+)'
        
        # Find all matches in the content
        matches = re.finditer(paper_pattern, content, re.DOTALL)
        
        for match in matches:
            title = match.group(1).strip()
            content = match.group(2).strip()
            arxiv_id = match.group(3).strip()
            arxiv_link = match.group(4).strip()
            upvotes = int(match.group(5))
            
            summaries.append({
                'title': title,
                'content': content,
                'arxiv_link': arxiv_link,
                'upvotes': upvotes
            })
        
        # Sort by upvotes
        summaries.sort(key=lambda x: x['upvotes'], reverse=True)
        return summaries

    def _generate_paper_html(self, paper: Dict) -> str:
        """Generate HTML for a single paper summary"""
        html = [
            '<div class="paper-summary">',
            f'    <div class="paper-title">',
            f'        <strong>{paper["title"]}</strong>',
            '    </div>',
            f'    <p>{paper["content"]}</p>',
            '    <div class="paper-footer">',
            f'        <a href="{paper["arxiv_link"]}" class="paper-link">Read Paper</a>',
            f'        <span class="paper-metrics">👍 {paper["upvotes"]} upvotes</span>',
            '    </div>',
            '</div>'
        ]
        return '\n'.join(html)

    def _generate_read_more_section(self, papers: List[Dict]) -> str:
        """Generate HTML for the read more section"""
        papers_list = '\n'.join([
            f'<li>{paper["title"]}</li>'
            for paper in papers[3:]
        ])
        
        html = f'''
        <div class="read-more-section">
            <h3 class="remaining-papers-header">More Breakthrough AI Research This Week</h3>
            <ul class="topics-list">
                {papers_list}
            </ul>
            <a href="https://shwetank-kumar.github.io/blog.html" class="cta-button">
                Read Full Summary
            </a>
        </div>
        '''
        return html
    
    def _generate_topics_list(self, papers: List[Dict]) -> str:
        return '\n'.join([
            f'<li>{paper["title"]}</li>'
            for paper in papers
        ])

    def _generate_papers_section(self, papers: List[Dict]) -> str:
        """Generate HTML for all papers including the Read More section"""
        # Generate HTML for top 3 papers with full summaries
        top_papers_html = '\n'.join([
            self._generate_paper_html(paper)
            for paper in papers[:3]
        ])
        
        # Generate count of remaining papers
        remaining_count = len(papers) - 3
        
        # Add "Read More" section if there are more papers
        if remaining_count > 0:
            read_more_html = f'''
                <div class="read-more-section">
                    <h3 class="remaining-papers-header">+ {remaining_count} More Exciting Papers! 🎉</h3>
                    <ul class="topics-list">
                        {self._generate_topics_list(papers[3:])}
                    </ul>
                    <div style="margin-top: 2rem;">
                        <a href="https://shwetank-kumar.github.io/blog.html" 
                        class="cta-button">
                            Read Full Research Summary
                        </a>
                    </div>
                </div>'''
            return top_papers_html + '\n' + read_more_html
            
        return top_papers_html

    def generate_newsletter(self, post_path: Path) -> str:
        content = post_path.read_text()
        metadata, content = self._parse_frontmatter(content)
        
        # Clean up the content
        content = re.sub(r'<iframe.*?</iframe>\s*', '', content, flags=re.DOTALL)
        
        # Extract papers
        papers = self._extract_paper_summaries(content)
        
        # Media and newsletter sections
        content_sections = f'''
        <div class="media-content">
            <h3>🎧 Listen to This Week's Summary</h3>
            <p>Prefer to listen? Check out our audio summary:</p>
            <a href="https://podcasters.spotify.com/pod/show/shwetankkumar" class="cta-button">
                Listen on Spotify
            </a>
        </div>

        <div class="newsletter-form">
            <h4>Never Miss an AI Afterhours Research Update</h4>
            <p>Get weekly summaries delivered straight to your inbox</p>
            <a href="https://shwetank-kumar.github.io/blog.html" class="cta-button">
                Subscribe Now
            </a>
        </div>'''

        # Generate paper summaries
        paper_summaries = ''
        if papers:
            paper_summaries = '\n'.join([self._generate_paper_html(paper) for paper in papers[:3]])
            
            # Add read more section if there are more papers
            if len(papers) > 3:
                paper_summaries += '\n' + self._generate_read_more_section(papers)
        
        categories_html = '\n'.join([
            f'<span class="category">{cat}</span>'
            for cat in metadata.get('categories', [])
        ])
        
        title = metadata.get('title', 'AI Afterhours')
        emoji = '🌙' if '🌙' in title else ''
        series_title = 'AI Afterhours'
        subtitle = f"Top AI Papers: {metadata.get('date', '')}"
        
        newsletter_html = self.html_template.format(
            title=title,
            emoji=emoji,
            series_title=series_title,
            subtitle=subtitle,
            categories=categories_html,
            introduction=content_sections,
            paper_summaries=paper_summaries
        )
        
        return newsletter_html

def main():
    import sys
    import traceback
    
    if len(sys.argv) != 2:
        print("Usage: python newsletter_generator.py relative/path/to/post.qmd")
        sys.exit(1)
    
    cwd = Path.cwd()
    post_path = cwd / sys.argv[1]
    
    if not post_path.exists():
        print(f"Error: File {post_path} does not exist")
        print(f"Current working directory: {cwd}")
        sys.exit(1)
    
    try:
        generator = NewsletterGenerator()
        newsletter_html = generator.generate_newsletter(post_path)
        output_path = post_path.parent / 'newsletter.html'
        output_path.write_text(newsletter_html)
        relative_output = output_path.relative_to(cwd)
        print(f"Newsletter generated and saved to: {relative_output}")
    except Exception as e:
        print(f"Error generating newsletter: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()