# 安装

## 通过包管理器安装

### npm

```bash
npm install WebInferjs
```

### yarn

```bash
yarn add WebInferjs
```

### pnpm

```bash
pnpm add WebInferjs
```

## 通过 CDN 使用

```html
<script type="module">
  import * as WebInfer from 'https://cdn.jsdelivr.net/npm/WebInferjs/dist/WebInfer.browser.min.js';
  
  // 使用 WebInfer
  const pipeline = await WebInfer.pipeline('text-classification');
</script>
```

## 浏览器兼容性

| 浏览器 | WebGPU | WebNN | WASM |
|--------|--------|-------|------|
| Chrome 113+ | ✅ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox 118+ | ⚠️ | ❌ | ✅ |
| Safari 17+ | ⚠️ | ❌ | ✅ |

## TypeScript 支持

WebInfer.js 使用 TypeScript 编写，提供完整的类型定义：

```typescript
import { pipeline, WebInferTensor, Tokenizer } from 'WebInferjs';
import type { PipelineOptions, TextClassificationResult } from 'WebInferjs';
```

## 下一步

- [快速入门](./quickstart.md)
- [核心概念](./concepts.md)
