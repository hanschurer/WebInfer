/**
 * WebInfer.js - Inference Scheduler
 *
 * Task scheduler for managing concurrent inference execution.
 * Supports priority queues, model-level isolation, and batch processing.
 */
import { WebInferError, ErrorCodes, } from './types.js';
// ============================================================================
// Task Implementation
// ============================================================================
/**
 * Internal task implementation
 */
class Task {
    id;
    modelId;
    priority;
    createdAt;
    _status = 'pending';
    _startedAt;
    _completedAt;
    _result;
    _error;
    _executor;
    _resolvers = [];
    _cancelled = false;
    constructor(id, modelId, priority, executor) {
        this.id = id;
        this.modelId = modelId;
        this.priority = priority;
        this.createdAt = Date.now();
        this._executor = executor;
    }
    get status() {
        return this._status;
    }
    get startedAt() {
        return this._startedAt;
    }
    get completedAt() {
        return this._completedAt;
    }
    get result() {
        return this._result;
    }
    get error() {
        return this._error;
    }
    /**
     * Cancel the task
     */
    cancel() {
        if (this._status === 'pending') {
            this._cancelled = true;
            this._status = 'cancelled';
            this._completedAt = Date.now();
            const cancelError = new WebInferError('Task was cancelled', ErrorCodes.INFERENCE_CANCELLED, { taskId: this.id });
            for (const { reject } of this._resolvers) {
                reject(cancelError);
            }
            this._resolvers = [];
        }
    }
    /**
     * Wait for task completion
     */
    wait() {
        if (this._status === 'completed') {
            return Promise.resolve(this._result);
        }
        if (this._status === 'failed') {
            return Promise.reject(this._error);
        }
        if (this._status === 'cancelled') {
            return Promise.reject(new WebInferError('Task was cancelled', ErrorCodes.INFERENCE_CANCELLED, { taskId: this.id }));
        }
        return new Promise((resolve, reject) => {
            this._resolvers.push({ resolve, reject });
        });
    }
    /**
     * Execute the task
     */
    async execute() {
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
        }
        catch (err) {
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
const PRIORITY_ORDER = {
    critical: 0,
    high: 1,
    normal: 2,
    low: 3,
};
/**
 * Priority queue for tasks
 */
class PriorityQueue {
    items = [];
    get length() {
        return this.items.length;
    }
    isEmpty() {
        return this.items.length === 0;
    }
    /**
     * Add item to queue with priority ordering
     */
    enqueue(item) {
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
    dequeue() {
        return this.items.shift();
    }
    /**
     * Peek at highest priority item without removing
     */
    peek() {
        return this.items[0];
    }
    /**
     * Remove a specific item by ID
     */
    remove(id) {
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
    getAll() {
        return [...this.items];
    }
    /**
     * Clear the queue
     */
    clear() {
        this.items = [];
    }
}
// ============================================================================
// Batch Collector
// ============================================================================
/**
 * Collects tasks for batch processing
 */
class BatchCollector {
    tasks = [];
    timer = null;
    maxSize;
    timeout;
    onBatch;
    constructor(maxSize, timeout, onBatch) {
        this.maxSize = maxSize;
        this.timeout = timeout;
        this.onBatch = onBatch;
    }
    add(task) {
        this.tasks.push(task);
        if (this.tasks.length >= this.maxSize) {
            this.flush();
        }
        else if (!this.timer) {
            this.timer = setTimeout(() => this.flush(), this.timeout);
        }
    }
    flush() {
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
    clear() {
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
function generateTaskId() {
    return `task_${++taskIdCounter}_${Date.now().toString(36)}`;
}
/**
 * Default scheduler options
 */
const DEFAULT_OPTIONS = {
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
    options;
    queues = new Map();
    runningTasks = new Map();
    allTasks = new Map();
    batchers = new Map();
    listeners = new Map();
    globalRunningCount = 0;
    isProcessing = false;
    disposed = false;
    constructor(options = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options };
    }
    /**
     * Get or create queue for a model
     */
    getQueue(modelId) {
        let queue = this.queues.get(modelId);
        if (!queue) {
            queue = new PriorityQueue();
            this.queues.set(modelId, queue);
        }
        return queue;
    }
    /**
     * Get or create running set for a model
     */
    getRunningSet(modelId) {
        let running = this.runningTasks.get(modelId);
        if (!running) {
            running = new Set();
            this.runningTasks.set(modelId, running);
        }
        return running;
    }
    /**
     * Check if we can start a new task for a model
     */
    canStartTask(modelId) {
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
    async processQueue() {
        if (this.isProcessing || this.disposed) {
            return;
        }
        this.isProcessing = true;
        try {
            // Find tasks that can be started
            const tasksToStart = [];
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
            await Promise.all(tasksToStart.map(async (task) => {
                this.emit('inference:start', { taskId: task.id, modelId: task.modelId });
                try {
                    await task.execute();
                    this.emit('inference:complete', {
                        taskId: task.id,
                        modelId: task.modelId,
                        duration: (task.completedAt ?? 0) - (task.startedAt ?? 0),
                    });
                }
                catch (error) {
                    this.emit('inference:error', {
                        taskId: task.id,
                        modelId: task.modelId,
                        error,
                    });
                }
                finally {
                    // Clean up
                    const running = this.runningTasks.get(task.modelId);
                    if (running) {
                        running.delete(task.id);
                    }
                    this.globalRunningCount--;
                }
            }));
        }
        finally {
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
    schedule(modelId, executor, priority = 'normal') {
        if (this.disposed) {
            throw new WebInferError('Scheduler has been disposed', ErrorCodes.RUNTIME_NOT_INITIALIZED);
        }
        const task = new Task(generateTaskId(), modelId, priority, executor);
        this.allTasks.set(task.id, task);
        // Add to queue
        const queue = this.getQueue(modelId);
        queue.enqueue(task);
        // Trigger processing
        this.processQueue();
        return task;
    }
    /**
     * Schedule with timeout
     */
    scheduleWithTimeout(modelId, executor, timeout = this.options.defaultTimeout, priority = 'normal') {
        const timeoutExecutor = () => {
            return new Promise((resolve, reject) => {
                const timer = setTimeout(() => {
                    reject(new WebInferError(`Task timed out after ${timeout}ms`, ErrorCodes.INFERENCE_TIMEOUT, { timeout }));
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
    async scheduleAll(tasks) {
        const scheduledTasks = tasks.map(({ modelId, executor, priority }) => this.schedule(modelId, executor, priority));
        return Promise.all(scheduledTasks.map(task => task.wait()));
    }
    /**
     * Get task by ID
     */
    getTask(taskId) {
        return this.allTasks.get(taskId);
    }
    /**
     * Cancel a task
     */
    cancelTask(taskId) {
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
    cancelAllForModel(modelId) {
        const queue = this.queues.get(modelId);
        if (!queue)
            return 0;
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
    getStats() {
        const stats = {
            totalTasks: this.allTasks.size,
            pendingTasks: 0,
            runningTasks: 0,
            completedTasks: 0,
            failedTasks: 0,
            cancelledTasks: 0,
            queuedByModel: {},
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
    on(event, listener) {
        let listeners = this.listeners.get(event);
        if (!listeners) {
            listeners = new Set();
            this.listeners.set(event, listeners);
        }
        listeners.add(listener);
    }
    /**
     * Remove event listener
     */
    off(event, listener) {
        const listeners = this.listeners.get(event);
        if (listeners) {
            listeners.delete(listener);
        }
    }
    /**
     * Emit event
     */
    emit(type, data) {
        const event = {
            type,
            timestamp: Date.now(),
            data,
        };
        const listeners = this.listeners.get(type);
        if (listeners) {
            for (const listener of listeners) {
                try {
                    listener(event);
                }
                catch (error) {
                    console.error('Error in event listener:', error);
                }
            }
        }
    }
    /**
     * Clear completed/failed/cancelled tasks from history
     */
    clearHistory() {
        for (const [taskId, task] of this.allTasks) {
            if (task.status === 'completed' ||
                task.status === 'failed' ||
                task.status === 'cancelled') {
                this.allTasks.delete(taskId);
            }
        }
    }
    /**
     * Dispose the scheduler
     */
    dispose() {
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
let globalScheduler = null;
/**
 * Get the global scheduler instance
 */
export function getScheduler() {
    if (!globalScheduler) {
        globalScheduler = new InferenceScheduler();
    }
    return globalScheduler;
}
/**
 * Set the global scheduler instance
 */
export function setScheduler(scheduler) {
    if (globalScheduler) {
        globalScheduler.dispose();
    }
    globalScheduler = scheduler;
}
/**
 * Configure the global scheduler
 */
export function configureScheduler(options) {
    setScheduler(new InferenceScheduler(options));
}
//# sourceMappingURL=scheduler.js.map