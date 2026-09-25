/** The catalog. The home page and the prev/next pager read from here. */

import type { ExplainerMeta } from '../explainer/types';

/** Where the companion blog lives. */
export const BLOG_ORIGIN = 'https://nbharathik.github.io/blogs';

export function blogUrl(path: string): string {
  const clean = path.startsWith('/') ? path : '/' + path;
  return BLOG_ORIGIN + clean;
}

/** Every explainer. `status: 'planned'` entries render as the roadmap. */
export const EXPLAINERS: ExplainerMeta[] = [
  {
    slug: 'linear-regression',
    title: 'Linear regression',
    tagline: 'Fit a straight line by walking downhill on the error',
    summary:
      'One input, one output, one line. Linear regression is the smallest model worth calling a model, and everything harder borrows its machinery: a loss to minimise, a gradient to follow, a learning rate to get wrong.',
    status: 'ready',
    difficulty: 'gentle',
    concepts: ['Least squares', 'Gradient descent', 'Learning rate', 'R²'],
    blogPath: '/linear-regression/',
    blogStatus: 'live',
    order: 1,
    symbol: '∕',
  },
  {
    slug: 'polynomial-regression',
    title: 'Polynomial regression',
    tagline: 'Bend the line until it fits the noise, then learn to stop',
    summary:
      'Feed a straight-line model the powers of x and it can bend. Step the degree up and watch the training error fall forever while the held-out error turns back up; refit on fresh noise fifty times and see the same curve split into bias and variance.',
    status: 'ready',
    difficulty: 'core',
    concepts: ['Polynomial features', 'Train vs test error', 'Bias-variance', 'Overfitting'],
    blogPath: '/polynomial-regression/',
    blogStatus: 'live',
    order: 2,
    symbol: '∿',
  },
  {
    slug: 'regularization',
    title: 'Ridge and lasso',
    tagline: 'Pay a price for large weights',
    summary:
      'Ten powers of x bend a curve through the noise; a penalty on the weights straightens it out. Ridge shrinks every weight, the lasso drives the useless ones to exactly zero, and the two-weight picture shows why: the squared-error rings meet the L1 diamond at a corner and the L2 disc anywhere.',
    status: 'ready',
    difficulty: 'core',
    concepts: ['L1 and L2', 'Shrinkage', 'Diamond and disc', 'Feature selection', 'Elastic net'],
    blogPath: '/regularization-ridge-lasso/',
    blogStatus: 'live',
    order: 3,
    symbol: 'λ',
  },
  {
    slug: 'logistic-regression',
    title: 'Logistic regression',
    tagline: 'Turn a straight line into a probability',
    summary:
      'Take the linear score, squash it through a sigmoid, and you have a classifier that says how sure it is. The boundary is still a line, what changes is what you do with the distance from it.',
    status: 'ready',
    difficulty: 'core',
    concepts: ['Sigmoid', 'Cross-entropy', 'Decision boundary', 'Thresholds'],
    blogPath: '/logistic-regression/',
    blogStatus: 'live',
    order: 4,
    symbol: 'S',
  },
  {
    slug: 'knn',
    title: 'k-nearest neighbours',
    tagline: 'No training at all, just look at who is nearby',
    summary:
      'k-NN keeps every training point and answers a query by voting among the k closest. There is no model to fit, which makes k and the distance metric the whole story.',
    status: 'ready',
    difficulty: 'gentle',
    concepts: ['Distance metrics', 'Voting', 'Bias-variance', 'Voronoi cells'],
    blogPath: '/knn/',
    blogStatus: 'live',
    order: 5,
    symbol: '◈',
  },
  {
    slug: 'decision-tree',
    title: 'Decision trees',
    tagline: 'Ask the one question that separates the data best, then repeat',
    summary:
      'A tree is a sequence of yes/no questions chosen greedily: at each node, pick the split that leaves the purest children. Grow it far enough and it will memorise the noise, which is exactly what makes pruning interesting.',
    status: 'ready',
    difficulty: 'core',
    concepts: ['Gini & entropy', 'Information gain', 'Greedy splits', 'Overfitting'],
    blogPath: '/decision-trees/',
    blogStatus: 'draft',
    order: 6,
    symbol: '⌥',
  },
  {
    slug: 'k-means',
    title: 'k-means clustering',
    tagline: 'Alternate between guessing groups and re‑centring them',
    summary:
      'Two steps, repeated: assign every point to its nearest centre, then move each centre to the middle of what it caught. It converges in a handful of rounds, and where it converges depends entirely on where it started.',
    status: 'ready',
    difficulty: 'gentle',
    concepts: ['Lloyd’s algorithm', 'Inertia', 'k-means++', 'Elbow method'],
    blogPath: '/k-means-clustering/',
    blogStatus: 'draft',
    order: 7,
    symbol: '◌',
  },
  {
    slug: 'neural-network',
    title: 'Neural networks',
    tagline: 'Stack neurons until the boundary can bend',
    summary:
      'A neural network is a pile of logistic regressions with a non-linearity between the layers. Watch each hidden unit carve out its own half-plane, then watch the output layer combine them into a boundary no single line could draw.',
    status: 'ready',
    difficulty: 'deep',
    concepts: ['Hidden layers', 'Activations', 'Backpropagation', 'Capacity'],
    blogPath: '/perceptron-mlp/',
    blogStatus: 'live',
    order: 8,
    symbol: '⋈',
  },
  {
    slug: 'activation-functions',
    title: 'Activations and losses',
    tagline: 'The two choices that decide what a network can express',
    summary:
      'Eight activations and six losses inside a live network: every unit drawn as its own curve with the values it sees, the output as the loss with the batch on it, and the gradient each layer receives. Watch the activation bend the input plane layer by layer, turn the output surface in three dimensions, and see why a stack with no activation is still logistic regression, why sigmoids vanish, ReLUs die and the hinge stops caring.',
    status: 'ready',
    difficulty: 'core',
    concepts: ['ReLU and friends', 'Bending the plane', 'Vanishing gradients', 'Cross-entropy', 'Softmax'],
    blogPath: '/activation-functions/',
    blogStatus: 'live',
    order: 9,
    symbol: 'ƒ',
  },
  {
    slug: 'cnn',
    title: 'Convolutional networks',
    tagline: 'Slide a small filter across the image and learn what it should look for',
    summary:
      'A handful of 3 × 3 kernels slide over a 16 × 16 glyph, ReLU keeps the matches, pooling forgives a shift, and a dense head reads the maps. Follow one image through every layer in 2D or 3D, slide a kernel over it pixel by pixel, and compare with a dense layer on raw pixels.',
    status: 'ready',
    difficulty: 'deep',
    concepts: ['Convolution', 'Receptive field', 'Pooling', 'Feature maps', 'Softmax'],
    blogPath: '/cnn/',
    blogStatus: 'draft',
    order: 10,
    symbol: '▦',
  },

  /* ---- roadmap ---- */
  {
    slug: 'svm',
    title: 'Support vector machines',
    tagline: 'Find the widest possible street between the classes',
    summary: '',
    status: 'planned',
    difficulty: 'deep',
    concepts: ['Margins', 'Kernels'],
    blogPath: '/svm/',
    blogStatus: 'draft',
    order: 11,
    symbol: '⫽',
  },
  {
    slug: 'naive-bayes',
    title: 'Naive Bayes',
    tagline: 'Assume the features are independent and see how far it gets you',
    summary: '',
    status: 'planned',
    difficulty: 'core',
    concepts: ['Priors', 'Likelihood'],
    blogPath: '/naive-bayes/',
    blogStatus: 'draft',
    order: 12,
    symbol: 'β',
  },
  {
    slug: 'random-forest',
    title: 'Random forests',
    tagline: 'Average many overfit trees into one that is not',
    summary: '',
    status: 'planned',
    difficulty: 'core',
    concepts: ['Bagging', 'Feature sampling'],
    blogPath: '/random-forests/',
    blogStatus: 'draft',
    order: 13,
    symbol: '⋔',
  },
  {
    slug: 'boosting',
    title: 'Gradient boosting',
    tagline: 'Each tree fixes what the last one got wrong',
    summary: '',
    status: 'planned',
    difficulty: 'deep',
    concepts: ['Residuals', 'Shrinkage'],
    blogPath: '/boosting/',
    blogStatus: 'draft',
    order: 14,
    symbol: '⇗',
  },
  {
    slug: 'pca',
    title: 'Principal component analysis',
    tagline: 'Rotate the data until the interesting direction is first',
    summary: '',
    status: 'planned',
    difficulty: 'core',
    concepts: ['Variance', 'Eigenvectors'],
    blogPath: '/pca/',
    blogStatus: 'draft',
    order: 15,
    symbol: '⤢',
  },
  {
    slug: 'dbscan',
    title: 'DBSCAN',
    tagline: 'Clusters are dense regions, not round blobs',
    summary: '',
    status: 'planned',
    difficulty: 'core',
    concepts: ['Density', 'Noise points'],
    blogPath: '/dbscan-hierarchical-clustering/',
    blogStatus: 'draft',
    order: 16,
    symbol: '◍',
  },
  {
    slug: 'transformers',
    title: 'Attention and transformers',
    tagline: 'Let every token look at every other and decide what matters',
    summary: '',
    status: 'planned',
    difficulty: 'deep',
    concepts: ['Self-attention', 'Positional encoding'],
    blogPath: '/transformers/',
    blogStatus: 'draft',
    order: 17,
    symbol: '⇶',
  },
  {
    slug: 'autoencoder',
    title: 'Autoencoders',
    tagline: 'Squeeze the data through a bottleneck and rebuild it',
    summary: '',
    status: 'planned',
    difficulty: 'core',
    concepts: ['Bottleneck', 'Reconstruction'],
    blogPath: '/autoencoders/',
    blogStatus: 'draft',
    order: 18,
    symbol: '⧖',
  },
  {
    slug: 'rnn',
    title: 'Recurrent networks',
    tagline: 'Carry a memory from one step of the sequence to the next',
    summary: '',
    status: 'planned',
    difficulty: 'deep',
    concepts: ['Hidden state', 'LSTM'],
    blogPath: '/rnn/',
    blogStatus: 'draft',
    order: 19,
    symbol: '↻',
  },
  {
    slug: 'dropout-batchnorm',
    title: 'Dropout and batch norm',
    tagline: 'The two training tricks, on the same small network',
    summary: '',
    status: 'planned',
    difficulty: 'core',
    concepts: ['Regularisation', 'Normalisation'],
    blogPath: '/dropout-batchnorm/',
    blogStatus: 'draft',
    order: 20,
    symbol: '⁝',
  },
];

export const READY_EXPLAINERS = EXPLAINERS.filter((e) => e.status === 'ready').sort(
  (a, b) => a.order - b.order,
);

export function getExplainer(slug: string): ExplainerMeta | undefined {
  return EXPLAINERS.find((e) => e.slug === slug);
}

/** Neighbours in the reading order, used by the page footer. */
export function getNeighbours(slug: string): {
  prev?: ExplainerMeta;
  next?: ExplainerMeta;
} {
  const index = READY_EXPLAINERS.findIndex((e) => e.slug === slug);
  if (index < 0) return {};
  return {
    prev: index > 0 ? READY_EXPLAINERS[index - 1] : undefined,
    next: index < READY_EXPLAINERS.length - 1 ? READY_EXPLAINERS[index + 1] : undefined,
  };
}
