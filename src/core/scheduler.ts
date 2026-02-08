/**
 * WebInfer.js - Inference Scheduler
 * 
 * Task scheduler for managing concurrent inference execution.
 * Supports priority queues, model-level isolation, and batch processing.
 */

import {
  InferenceTask,
  TaskPriority,
  TaskStatus,
  SchedulerOptions,
  WebInferError,
  ErrorCodes,
  EventType,
  EventListener,
  WebInferEvent,
} from './types.js';

// ============================================================================
// Task Implementation
// ============================================================================

/**
 * Internal task implementation
 */
class Task<T = unknown> implements InferenceTask<T> {
  readonly id: string;
  readonly modelId: string;
  readonly priority: TaskPriority;
  readonly createdAt: number;
  
  private _status: TaskStatus = 'pending';
  private _startedAt?: number;
  private _completedAt?: number;
  private _result?: T;
  private _error?: Error;
  private _executor: () => Promise<T>;
  private _resolvers: Array<{
    resolve: (value: T) => void;
    reject: (error: Error) => void;
  }> = [];
  private _cancelled = false;

  constructor(
    id: string,
    modelId: string,
    priority: TaskPriority,
    executor: () => Promise<T>
  ) {
    this.id = id;
    this.modelId = modelId;
    this.priority = priority;
    this.createdAt = Date.now();
    this._executor = executor;
  }

  get status(): TaskStatus {
    return this._status;
  }

  get startedAt(): number | undefined {
    return this._startedAt;
  }

  get completedAt(): number | undefined {
    return this._completedAt;
  }

  get result(): T | undefined {
    return this._result;
  }

  get error(): Error | undefined {
    return this._error;
  }

  /**
   * Cancel the task
   */
  cancel(): void {
    if (this._status === 'pending') {
      this._cancelled = true;
      this._status = 'cancelled';
      this._completedAt = Date.now();
      
      const cancelError = new WebInferError(
        'Task was cancelled',
        ErrorCodes.INFERENCE_CANCELLED,
        { taskId: this.id }
      );
      
      for (const { reject } of this._resolvers) {
        reject(cancelError);
      }
      this._resolvers = [];
    }
  }

  /**
   * Wait for task completion
   */
  wait(): Promise<T> {
    if (this._status === 'completed') {
      return Promise.resolve(this._result as T);
    }
    
    if (this._status === 'failed') {
      return Promise.reject(this._error);
    }
    
    if (this._status === 'cancelled') {
      return Promise.reject(new WebInferError(
        'Task was cancelled',
        ErrorCodes.INFERENCE_CANCELLED,
        { taskId: this.id }
      ));
    }

    return new Promise<T>((resolve, reject) => {
      this._resolvers.push({ resolve, reject });
    });
  }

  /**
   * Execute the task
   */
  async execute(): Promise<void> {
    if (this._cancelled) {
      return;
    }

    this._status = 'running';
    this._startedAt = Date.now();

    try {
      this._result = await this._executor();
      this._status = 'completed';
      this._completedAt = Date.now();
      
      for (const { resolve } of this._resolvers) {
        resolve(this._result);
      }
    } catch (err) {
      this._error = err instanceof Error ? err : new Error(String(err));
      this._status = 'failed';
      this._completedAt = Date.now();
      
      for (const { reject } of this._resolvers) {
        reject(this._error);
      }
    }
    
    this._resolvers = [];
  }
}

// ============================================================================
// Priority Queue Implementation
// ============================================================================

/**
 * Priority mapping for comparison
 */
const PRIORITY_ORDER: Record<TaskPriority, number> = {
  critical: 0,
  high: 1,
  normal: 2,
  low: 3,
};

/**
 * Priority queue for tasks
 */
class PriorityQueue<T extends Task> {
  private items: T[] = [];

  get length(): number {
    return this.items.length;
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  /**
   * Add item to queue with priority ordering
   */
  enqueue(item: T): void {
    let inserted = false;
    
    for (let i = 0; i < this.items.length; i++) {
      const currentItem = this.items[i];
      if (currentItem && PRIORITY_ORDER[item.priority] < PRIORITY_ORDER[currentItem.priority]) {
        this.items.splice(i, 0, item);
        inserted = true;
        break;
      }
    }
    
    if (!inserted) {
      this.items.push(item);
    }
  }

  /**
   * Remove and return highest priority item
   */
  dequeue(): T | undefined {
    return this.items.shift();
  }

  /**
   * Peek at highest priority item without removing
   */
  peek(): T | undefined {
    return this.items[0];
  }

  /**
   * Remove a specific item by ID
   */
  remove(id: string): T | undefined {
    const index = this.items.findIndex(item => item.id === id);
    if (index !== -1) {
      const [removed] = this.items.splice(index, 1);
      return removed;
    }
    return undefined;
  }

  /**
   * Get all items
   */
  getAll(): T[] {
    return [...this.items];
  }

  /**
   * Clear the queue
   */
  clear(): void {
    this.items = [];
  }
}

// ============================================================================
// Batch Collector
// ============================================================================

/**
 * Collects tasks for batch processing
 */
class BatchCollector<T> {
  private tasks: Task<T>[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private readonly maxSize: number;
  private readonly timeout: number;
  private readonly onBatch: (tasks: Task<T>[]) => void;

  constructor(
    maxSize: number,
    timeout: number,
    onBatch: (tasks: Task<T>[]) => void
  ) {
    this.maxSize = maxSize;
    this.timeout = timeout;
    this.onBatch = onBatch;
  }

  add(task: Task<T>): void {
    this.tasks.push(task);

    if (this.tasks.length >= this.maxSize) {
      this.flush();
    } else if (!this.timer) {
      this.timer = setTimeout(() => this.flush(), this.timeout);
    }
  }

  flush(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    if (this.tasks.length > 0) {
      const batch = this.tasks;
      this.tasks = [];
      this.onBatch(batch);
    }
  }

  clear(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.tasks = [];
  }
}

// ============================================================================
// Inference Scheduler
// ============================================================================

// Counter for task IDs
let taskIdCounter = 0;

/**
 * Generate unique task ID
 */
function generateTaskId(): string {
  return `task_${++taskIdCounter}_${Date.now().toString(36)}`;
}

/**
 * Default scheduler options
 */
const DEFAULT_OPTIONS: Required<SchedulerOptions> = {
  maxConcurrentTasks: 4,
  maxConcurrentPerModel: 1,
  defaultTimeout: 30000,
  enableBatching: false,
  maxBatchSize: 32,
  batchTimeout: 50,
};

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
export class InferenceScheduler {
  private readonly options: Required<SchedulerOptions>;
  private readonly queues: Map<string, PriorityQueue<Task>> = new Map();
  private readonly runningTasks: Map<string, Set<string>> = new Map();
  private readonly allTasks: Map<string, Task> = new Map();
  private readonly batchers: Map<string, BatchCollector<unknown>> = new Map();
  private readonly listeners: Map<EventType, Set<EventListener>> = new Map();
  private globalRunningCount = 0;
  private isProcessing = false;
  private disposed = false;

  constructor(options: SchedulerOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Get or create queue for a model
   */
  private getQueue(modelId: string): PriorityQueue<Task> {
    let queue = this.queues.get(modelId);
    if (!queue) {
      queue = new PriorityQueue<Task>();
      this.queues.set(modelId, queue);
    }
    return queue;
  }

  /**
   * Get or create running set for a model
   */
  private getRunningSet(modelId: string): Set<string> {
    let running = this.runningTasks.get(modelId);
    if (!running) {
      running = new Set<string>();
      this.runningTasks.set(modelId, running);
    }
    return running;
  }

  /**
   * Check if we can start a new task for a model
   */
  private canStartTask(modelId: string): boolean {
    if (this.globalRunningCount >= this.options.maxConcurrentTasks) {
      return false;
    }

    const running = this.runningTasks.get(modelId);
    if (running && running.size >= this.options.maxConcurrentPerModel) {
      return false;
    }

    return true;
  }

  /**
   * Process pending tasks
   */
  private async processQueue(): Promise<void> {
    if (this.isProcessing || this.disposed) {
      return;
    }

    this.isProcessing = true;

    try {
      // Find tasks that can be started
      const tasksToStart: Task[] = [];

      for (const [modelId, queue] of this.queues) {
        while (!queue.isEmpty() && this.canStartTask(modelId)) {
          const task = queue.dequeue();
          if (task && task.status === 'pending') {
            tasksToStart.push(task);
            
            const running = this.getRunningSet(modelId);
            running.add(task.id);
            this.globalRunningCount++;
          }
        }
      }

      // Execute tasks concurrently
      await Promise.all(
        tasksToStart.map(async (task) => {
          this.emit('inference:start', { taskId: task.id, modelId: task.modelId });

          try {
            await task.execute();
            this.emit('inference:complete', {
              taskId: task.id,
              modelId: task.modelId,
              duration: (task.completedAt ?? 0) - (task.startedAt ?? 0),
            });
          } catch (error) {
            this.emit('inference:error', {
              taskId: task.id,
              modelId: task.modelId,
              error,
            });
          } finally {
            // Clean up
            const running = this.runningTasks.get(task.modelId);
            if (running) {
              running.delete(task.id);
            }
            this.globalRunningCount--;
          }
        })
      );
    } finally {
      this.isProcessing = false;
    }

    // Check if there are more tasks to process
    let hasPending = false;
    for (const queue of this.queues.values()) {
      if (!queue.isEmpty()) {
        hasPending = true;
        break;
      }
    }

    if (hasPending) {
      // Use setImmediate-like behavior for next tick processing
      setTimeout(() => this.processQueue(), 0);
    }
  }

  /**
   * Schedule a task for execution
   */
  schedule<T>(
    modelId: string,
    executor: () => Promise<T>,
    priority: TaskPriority = 'normal'
  ): InferenceTask<T> {
    if (this.disposed) {
      throw new WebInferError(
        'Scheduler has been disposed',
        ErrorCodes.RUNTIME_NOT_INITIALIZED
      );
    }

    const task = new Task<T>(
      generateTaskId(),
      modelId,
      priority,
      executor
    );

    this.allTasks.set(task.id, task as Task);

    // Add to queue
    const queue = this.getQueue(modelId);
    queue.enqueue(task as Task);

    // Trigger processing
    this.processQueue();

    return task;
  }

  /**
   * Schedule with timeout
   */
  scheduleWithTimeout<T>(
    modelId: string,
    executor: () => Promise<T>,
    timeout: number = this.options.defaultTimeout,
    priority: TaskPriority = 'normal'
  ): InferenceTask<T> {
    const timeoutExecutor = (): Promise<T> => {
      return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => {
          reject(new WebInferError(
            `Task timed out after ${timeout}ms`,
            ErrorCodes.INFERENCE_TIMEOUT,
            { timeout }
          ));
        }, timeout);

        executor()
          .then(result => {
            clearTimeout(timer);
            resolve(result);
          })
          .catch(error => {
            clearTimeout(timer);
            reject(error);
          });
      });
    };

    return this.schedule(modelId, timeoutExecutor, priority);
  }

  /**
   * Schedule multiple tasks and wait for all
   */
  async scheduleAll<T>(
    tasks: Array<{
      modelId: string;
      executor: () => Promise<T>;
      priority?: TaskPriority;
    }>
  ): Promise<T[]> {
    const scheduledTasks = tasks.map(({ modelId, executor, priority }) =>
      this.schedule<T>(modelId, executor, priority)
    );

    return Promise.all(scheduledTasks.map(task => task.wait()));
  }

  /**
   * Get task by ID
   */
  getTask(taskId: string): InferenceTask | undefined {
    return this.allTasks.get(taskId);
  }

  /**
   * Cancel a task
   */
  cancelTask(taskId: string): boolean {
    const task = this.allTasks.get(taskId);
    if (task && task.status === 'pending') {
      task.cancel();
      
      // Remove from queue
      for (const queue of this.queues.values()) {
        queue.remove(taskId);
      }
      
      return true;
    }
    return false;
  }

  /**
   * Cancel all tasks for a model
   */
  cancelAllForModel(modelId: string): number {
    const queue = this.queues.get(modelId);
    if (!queue) return 0;

    let cancelled = 0;
    for (const task of queue.getAll()) {
      if (task.status === 'pending') {
        task.cancel();
        cancelled++;
      }
    }
    queue.clear();
    
    return cancelled;
  }

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
  } {
    const stats = {
      totalTasks: this.allTasks.size,
      pendingTasks: 0,
      runningTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      cancelledTasks: 0,
      queuedByModel: {} as Record<string, number>,
    };

    for (const task of this.allTasks.values()) {
      switch (task.status) {
        case 'pending':
          stats.pendingTasks++;
          break;
        case 'running':
          stats.runningTasks++;
          break;
        case 'completed':
          stats.completedTasks++;
          break;
        case 'failed':
          stats.failedTasks++;
          break;
        case 'cancelled':
          stats.cancelledTasks++;
          break;
      }
    }

    for (const [modelId, queue] of this.queues) {
      stats.queuedByModel[modelId] = queue.length;
    }

    return stats;
  }

  /**
   * Add event listener
   */
  on<T = unknown>(event: EventType, listener: EventListener<T>): void {
    let listeners = this.listeners.get(event);
    if (!listeners) {
      listeners = new Set();
      this.listeners.set(event, listeners);
    }
    listeners.add(listener as EventListener);
  }

  /**
   * Remove event listener
   */
  off<T = unknown>(event: EventType, listener: EventListener<T>): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.delete(listener as EventListener);
    }
  }

  /**
   * Emit event
   */
  private emit<T>(type: EventType, data: T): void {
    const event: WebInferEvent<T> = {
      type,
      timestamp: Date.now(),
      data,
    };

    const listeners = this.listeners.get(type);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(event);
        } catch (error) {
          console.error('Error in event listener:', error);
        }
      }
    }
  }

  /**
   * Clear completed/failed/cancelled tasks from history
   */
  clearHistory(): void {
    for (const [taskId, task] of this.allTasks) {
      if (
        task.status === 'completed' ||
        task.status === 'failed' ||
        task.status === 'cancelled'
      ) {
        this.allTasks.delete(taskId);
      }
    }
  }

  /**
   * Dispose the scheduler
   */
  dispose(): void {
    this.disposed = true;

    // Cancel all pending tasks
    for (const queue of this.queues.values()) {
      for (const task of queue.getAll()) {
        task.cancel();
      }
      queue.clear();
    }

    // Clear batchers
    for (const batcher of this.batchers.values()) {
      batcher.clear();
    }

    this.queues.clear();
    this.runningTasks.clear();
    this.allTasks.clear();
    this.batchers.clear();
    this.listeners.clear();
  }
}

// ============================================================================
// Global Scheduler Instance
// ============================================================================

let globalScheduler: InferenceScheduler | null = null;

/**
 * Get the global scheduler instance
 */
export function getScheduler(): InferenceScheduler {
  if (!globalScheduler) {
    globalScheduler = new InferenceScheduler();
  }
  return globalScheduler;
}

/**
 * Set the global scheduler instance
 */
export function setScheduler(scheduler: InferenceScheduler): void {
  if (globalScheduler) {
    globalScheduler.dispose();
  }
  globalScheduler = scheduler;
}

/**
 * Configure the global scheduler
 */
export function configureScheduler(options: SchedulerOptions): void {
  setScheduler(new InferenceScheduler(options));
}
