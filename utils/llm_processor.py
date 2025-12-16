import yaml
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from utils.logger import get_logger

logger = get_logger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Load prompt templates
prompt_template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts')
template_env = Environment(loader=FileSystemLoader(prompt_template_path))

class LLMProcessor:
    """LLM Processor for knowledge integration"""
    
    def __init__(self):
        """Initialize LLM processor with configuration."""
        llm_config = config.get('llm', {})
        
        self.client = OpenAI(
            api_key=llm_config.get('api_key', ''),
            base_url=llm_config.get('base_url', 'https://api.openai.com/v1'),
        )
        
        self.model_name = llm_config.get('model_name', 'gpt-4o-mini')
        self.timeout = llm_config.get('timeout', 30)
        self.max_retries = llm_config.get('max_retries', 3)
        
        # Load prompt template
        self.template = template_env.get_template('knowledge_integration.yaml')
        
        # 定义类型列表
        self.types = [
            '区块链',
            '一般加密货币',
            '稳定币',
            '技术分析',
            '金融知识',
            '经济学知识',
            '政策解读',
            '交易策略与纪律',
        ]
        
        logger.info("LLM Processor initialized")
    
    def integrate_knowledge(self, content: str, source: str = "") -> Dict[str, Any]:
        """
        Integrate knowledge using LLM.
        
        Args:
            content (str): Content to be integrated
            source (str): Source of the content
            
        Returns:
            Dict[str, Any]: Integrated knowledge in structured format
        """
        try:
            # Prepare prompt
            system_prompt = self.template.module.system_prompt
            user_prompt = self.template.render(content=content, types=self.types)
            
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                timeout=self.timeout
            )
            
            # Parse response
            result_str = response.choices[0].message.content.strip()
            
            # 解析JSON结果
            result = json.loads(result_str)
            
            # 验证必要字段
            required_fields = ['content', 'type', 'timestamp', 'source']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # 验证类型是否在允许的范围内
            if result['type'] not in self.types:
                raise ValueError(f"Invalid type: {result['type']}")
            
            logger.info(f"Knowledge integrated successfully")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to integrate knowledge: {str(e)}")
            raise
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks using LLM integration.
        
        Args:
            chunks (List[Dict[str, Any]]): Chunks to be processed
            
        Returns:
            List[Dict[str, Any]]: Processed chunks
        """
        processed_chunks = []
        
        for chunk in chunks:
            try:
                # 获取原始内容和来源
                content = chunk.get('text', '')
                source = chunk.get('source', '')
                
                # 使用大模型整合知识
                integrated_result = self.integrate_knowledge(content, source)
                
                # 更新chunk信息
                processed_chunk = chunk.copy()
                processed_chunk['text'] = integrated_result['content']
                processed_chunk['domain'] = integrated_result['type']
                processed_chunk['timestamp'] = integrated_result['timestamp']
                processed_chunk['source'] = integrated_result['source']
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.warning(f"Failed to process chunk: {str(e)}. Using original chunk.")
                # 出错时使用原始chunk
                processed_chunks.append(chunk)
        
        return processed_chunks