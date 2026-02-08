/**
 * WebInfer.js - Inference Scheduler
 *
 * Task scheduler for managing concurrent inference execution.
 * Supports priority queues, model-level isolation, and batch processing.
 */
import { InferenceTask, TaskPriority, SchedulerOptions, EventType, EventListener } from './types.js';
/**
 * InferenceScheduler - Manages concurrent task execution
 *
 * Features:
 * - Priority-based task scheduling
 * - Model-level concurrency control
 * - Optional batch processing
 * - Task cancellation
 * - Event emission
 */
export declare class InferenceScheduler {
    private readonly options;
    private readonly queues;
    private readonly runningTasks;
    private readonly allTasks;
    private readonly batchers;
    private readonly listeners;
    private globalRunningCount;
    private isProcessing;
    private disposed;
    constructor(options?: SchedulerOptions);
    /**
     * Get or create queue for a model
     */
    private getQueue;
    /**
     * Get or create running set for a model
     */
    private getRunningSet;
    /**
     * Check if we can start a new task for a model
     */
    private canStartTask;
    /**
     * Process pending tasks
     */
    private processQueue;
    /**
     * Schedule a task for execution
     */
    schedule<T>(modelId: string, executor: () => Promise<T>, priority?: TaskPriority): InferenceTask<T>;
    /**
     * Schedule with timeout
     */
    scheduleWithTimeout<T>(modelId: string, executor: () => Promise<T>, timeout?: number, priority?: TaskPriority): InferenceTask<T>;
    /**
     * Schedule multiple tasks and wait for all
     */
    scheduleAll<T>(tasks: Array<{
        modelId: string;
        executor: () => Promise<T>;
        priority?: TaskPriority;
    }>): Promise<T[]>;
    /**
     * Get task by ID
     */
    getTask(taskId: string): InferenceTask | undefined;
    /**
     * Cancel a task
     */
    cancelTask(taskId: string): boolean;
    /**
     * Cancel all tasks for a model
     */
    cancelAllForModel(modelId: string): number;
    /**
     * Get statistics
     */
    getStats(): {
        totalTasks: number;
        pendingTasks: number;
        runningTasks: number;
        completedTasks: number;
        failedTasks: number;
        cancelledTasks: number;
        queuedByModel: Record<string, number>;
    };
    /**
     * Add event listener
     */
    on<T = unknown>(event: EventType, listener: EventListener<T>): void;
    /**
     * Remove event listener
     */
    off<T = unknown>(event: EventType, listener: EventListener<T>): void;
    /**
     * Emit event
     */
    private emit;
    /**
     * Clear completed/failed/cancelled tasks from history
     */
    clearHistory(): void;
    /**
     * Dispose the scheduler
     */
    dispose(): void;
}
/**
 * Get the global scheduler instance
 */
export declare function getScheduler(): InferenceScheduler;
/**
 * Set the global scheduler instance
 */
export declare function setScheduler(scheduler: InferenceScheduler): void;
/**
 * Configure the global scheduler
 */
export declare function configureScheduler(options: SchedulerOptions): void;
//# sourceMappingURL=scheduler.d.ts.map