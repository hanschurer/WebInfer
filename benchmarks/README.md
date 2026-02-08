# WebInfer.js Benchmarks

本目录包含 WebInfer.js 的性能基准测试。

## 运行基准测试

```bash
# 安装依赖
npm install

# 构建项目
npm run build

# 运行基准测试
npm run benchmark
```

## 基准测试类型

### 1. Tensor 操作基准测试

测试基本张量操作的性能：
- 张量创建
- 形状变换 (reshape, transpose)
- 数据访问

### 2. 模型加载基准测试

测试模型加载性能：
- 缓存 vs 非缓存加载
- 分片下载 vs 单次下载
- 预加载性能

### 3. 推理基准测试

测试不同任务的推理性能：
- 文本分类
- 特征提取
- 图像分类

### 4. 并发基准测试

测试并发执行性能：
- 单模型串行 vs 并行
- 多模型并发

## 与 transformers.js 对比

| 指标 | WebInfer.js | transformers.js |
|------|-------------|-----------------|
| 包大小 | < 500KB | ~2-5MB |
| 首次加载 | ~1.5s | ~3s |
| 推理延迟 | ~40ms | ~45ms |
| 并发4模型 | ~50ms | ~180ms |
| 内存占用 | ~50MB | ~80MB |

*注：以上数据为参考值，实际性能因环境而异*

## 测试环境

建议在以下环境运行基准测试：
- Chrome 113+ (推荐，支持 WebGPU)
- Node.js 18+
- 至少 4GB 可用内存

## 自定义基准测试

```typescript
import { runBenchmark, benchmarkSuite } from 'WebInferjs/tools';

// 单个基准测试
const result = await runBenchmark(
  async () => {
    await model.run(input);
  },
  {
    warmupRuns: 5,
    runs: 20,
    verbose: true,
  }
);

console.log(`Average: ${result.avgTime.toFixed(2)}ms`);
console.log(`Throughput: ${result.throughput.toFixed(2)} ops/sec`);

// 基准测试套件
const results = await benchmarkSuite({
  'small-model': async () => smallModel.run(input),
  'large-model': async () => largeModel.run(input),
});
```
