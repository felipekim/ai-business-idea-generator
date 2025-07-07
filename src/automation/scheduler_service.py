"""
Automated Scheduler Service
Handles automated idea generation on scheduled intervals
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ScheduleFrequency(Enum):
    """Schedule frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ScheduledTask:
    """Scheduled task configuration"""
    task_id: str
    name: str
    frequency: ScheduleFrequency
    target_ideas: int  # Number of ideas to generate
    quality_threshold: float  # Minimum quality score
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Calculate next run time"""
        if self.next_run is None:
            self.next_run = self._calculate_next_run()
    
    def _calculate_next_run(self) -> datetime:
        """Calculate next run time based on frequency"""
        now = datetime.utcnow()
        
        if self.frequency == ScheduleFrequency.DAILY:
            return now + timedelta(days=1)
        elif self.frequency == ScheduleFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif self.frequency == ScheduleFrequency.BIWEEKLY:
            return now + timedelta(weeks=2)
        elif self.frequency == ScheduleFrequency.MONTHLY:
            return now + timedelta(days=30)
        else:
            return now + timedelta(days=1)  # Default to daily

@dataclass
class AutomationConfig:
    """Automation configuration"""
    enabled: bool = True
    default_frequency: ScheduleFrequency = ScheduleFrequency.WEEKLY
    default_target_ideas: int = 7  # 5-10 range, default to 7
    quality_threshold: float = 7.0
    max_concurrent_tasks: int = 3
    retry_delay_minutes: int = 30
    notification_enabled: bool = True
    
    # Quality gates
    min_sources_per_idea: int = 5
    max_sources_per_idea: int = 8
    source_quality_threshold: float = 0.7
    
    # Performance settings
    batch_processing: bool = True
    cache_enabled: bool = True
    parallel_research: bool = True

class AutomatedSchedulerService:
    """Automated scheduler service for idea generation"""
    
    def __init__(self, config: Optional[AutomationConfig] = None):
        self.config = config or AutomationConfig()
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self._scheduler_thread = None
        
        logger.info("Automated Scheduler Service initialized")
    
    def add_task(self, task: ScheduledTask) -> bool:
        """Add a scheduled task"""
        try:
            self.tasks[task.task_id] = task
            logger.info(f"Added scheduled task: {task.name} ({task.frequency.value})")
            return True
        except Exception as e:
            logger.error(f"Error adding task {task.task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task"""
        try:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"Removed scheduled task: {task_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing task {task_id}: {e}")
            return False
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Automated scheduler started")
    
    def stop_scheduler(self):
        """Stop the automated scheduler"""
        self.running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Automated scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                # Check for tasks that need to run
                for task_id, task in self.tasks.items():
                    if (task.next_run <= current_time and 
                        task.status not in [TaskStatus.RUNNING, TaskStatus.CANCELLED]):
                        
                        # Submit task for execution
                        self.executor.submit(self._execute_task, task)
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Continue after error
    
    def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task"""
        logger.info(f"Executing task: {task.name}")
        
        try:
            task.status = TaskStatus.RUNNING
            task.last_run = datetime.utcnow()
            
            # Import here to avoid circular imports
            from src.automation.intelligent_discovery import IntelligentDiscoveryEngine
            from src.services.enhanced_ai_service import EnhancedAIService
            from src.services.market_research_service import MarketResearchService
            from src.services.financial_analysis_service import FinancialAnalysisService
            from src.evaluation.custom_scoring_engine import CustomScoringEngine
            
            # Initialize services
            discovery_engine = IntelligentDiscoveryEngine()
            ai_service = EnhancedAIService()
            market_service = MarketResearchService()
            financial_service = FinancialAnalysisService()
            scoring_engine = CustomScoringEngine()
            
            # Step 1: Discover trending topics and keywords
            logger.info("Discovering trending topics...")
            trending_topics = discovery_engine.discover_trending_opportunities()
            
            # Step 2: Generate ideas based on trends
            logger.info(f"Generating {task.target_ideas} business ideas...")
            generated_ideas = []
            
            for i in range(task.target_ideas):
                try:
                    # Select trending topic for this idea
                    topic = trending_topics[i % len(trending_topics)] if trending_topics else "AI automation"
                    
                    # Generate idea with context
                    ideas = ai_service.generate_enhanced_business_ideas(
                        count=1, 
                        context={"trending_topic": topic}
                    )
                    
                    if ideas:
                        idea = ideas[0]
                        
                        # Step 3: Conduct research and analysis
                        market_research = market_service.research_market_opportunity(idea)
                        financial_analysis = financial_service.analyze_financial_projections(idea)
                        
                        # Step 4: Score the idea
                        scores = scoring_engine.evaluate_business_idea(
                            research_data=idea,
                            idea_title=idea.get('name', f'Automated Idea {i+1}')
                        )
                        
                        # Step 5: Quality gate check
                        if scores.overall_score >= task.quality_threshold:
                            idea['market_research'] = market_research
                            idea['financial_analysis'] = financial_analysis
                            idea['custom_scores'] = scores.__dict__
                            idea['automation_metadata'] = {
                                'generated_by': 'automated_scheduler',
                                'task_id': task.task_id,
                                'trending_topic': topic,
                                'generation_timestamp': datetime.utcnow().isoformat()
                            }
                            generated_ideas.append(idea)
                            logger.info(f"Generated quality idea: {idea.get('name')} (Score: {scores.overall_score})")
                        else:
                            logger.info(f"Idea rejected due to low quality score: {scores.overall_score}")
                
                except Exception as e:
                    logger.error(f"Error generating idea {i+1}: {e}")
                    continue
            
            # Step 6: Store results
            task.results = {
                'generated_ideas': generated_ideas,
                'total_generated': len(generated_ideas),
                'target_ideas': task.target_ideas,
                'trending_topics': trending_topics,
                'execution_time': (datetime.utcnow() - task.last_run).total_seconds(),
                'quality_threshold': task.quality_threshold,
                'success_rate': len(generated_ideas) / task.target_ideas if task.target_ideas > 0 else 0
            }
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.next_run = task._calculate_next_run()
            task.error_count = 0  # Reset error count on success
            
            logger.info(f"Task completed successfully: {task.name}")
            logger.info(f"Generated {len(generated_ideas)}/{task.target_ideas} quality ideas")
            
        except Exception as e:
            logger.error(f"Error executing task {task.name}: {e}")
            task.status = TaskStatus.FAILED
            task.error_count += 1
            
            # Schedule retry if under max retries
            if task.error_count < task.max_retries:
                task.next_run = datetime.utcnow() + timedelta(minutes=self.config.retry_delay_minutes)
                task.status = TaskStatus.PENDING
                logger.info(f"Task will retry in {self.config.retry_delay_minutes} minutes")
            else:
                logger.error(f"Task {task.name} failed after {task.max_retries} retries")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'task_id': task.task_id,
            'name': task.name,
            'frequency': task.frequency.value,
            'status': task.status.value,
            'last_run': task.last_run.isoformat() if task.last_run else None,
            'next_run': task.next_run.isoformat() if task.next_run else None,
            'error_count': task.error_count,
            'results': task.results
        }
    
    def get_all_tasks_status(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
    
    def create_default_weekly_task(self) -> ScheduledTask:
        """Create default weekly automation task"""
        task = ScheduledTask(
            task_id="weekly_automation",
            name="Weekly Business Idea Generation",
            frequency=ScheduleFrequency.WEEKLY,
            target_ideas=self.config.default_target_ideas,
            quality_threshold=self.config.quality_threshold
        )
        
        self.add_task(task)
        return task
    
    def force_run_task(self, task_id: str) -> bool:
        """Force immediate execution of a task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.RUNNING:
            logger.warning(f"Task {task_id} is already running")
            return False
        
        # Submit for immediate execution
        self.executor.submit(self._execute_task, task)
        logger.info(f"Force-started task: {task.name}")
        return True

# Global scheduler instance
_scheduler_instance = None

def get_scheduler() -> AutomatedSchedulerService:
    """Get global scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutomatedSchedulerService()
    return _scheduler_instance

def initialize_automation():
    """Initialize automation with default settings"""
    scheduler = get_scheduler()
    
    # Create default weekly task
    default_task = scheduler.create_default_weekly_task()
    
    # Start scheduler
    scheduler.start_scheduler()
    
    logger.info("Automation initialized with weekly task")
    return scheduler

