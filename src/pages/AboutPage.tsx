import { BLOG_ORIGIN } from '../explainers/registry';
import { useDocumentTitle } from '../lib/useDocumentTitle';

export function AboutPage() {
  useDocumentTitle('About');
  return (
    <div className="mlx-prose-page">
      <header className="mlx-prose-page__head">
        <h1>About</h1>
        <p>
          Interactive Machine Learning is a collection of hands-on visual
          explainers designed to make machine-learning algorithms easier to
          understand. Each model runs live in your browser, allowing you to
          change the data, adjust parameters, and immediately see how the
          algorithm responds.
        </p>
        <p>
          The project is the interactive companion to{' '}
          <a href={BLOG_ORIGIN} target="_blank" rel="noreferrer">
            B2logs
          </a>
          . The articles explain the mathematical foundations; this platform
          lets you experiment with the concepts in practice.
        </p>
      </header>

      <section>
        <h2>Why this project?</h2>
        <p>
          Equations describe how an algorithm works, but intuition develops
          through interaction. Adjust the parameters, modify the data, and
          observe how each decision changes the model. These are not predefined
          animations. Every explainer runs an actual implementation of the
          algorithm in your browser. The models can learn, adapt, and fail just
          as they do in practice. Set the learning rate too high, for example,
          and you can watch gradient descent diverge in real time.
        </p>
        <p>
          Everything runs locally in your browser and no data is uploaded.
          Experiments use seeded randomness, so shared configurations can
          reproduce the same data, settings, and results.
        </p>
      </section>

      <section>
        <h2>Keyboard shortcuts</h2>
        <ul className="mlx-keys">
          <li>
            <kbd>Space</kbd> Play or pause
          </li>
          <li>
            <kbd>→</kbd> Advance one step
          </li>
          <li>
            <kbd>Shift</kbd> + <kbd>→</kbd> Advance ten steps
          </li>
          <li>
            <kbd>R</kbd> Reset the current run
          </li>
          <li>
            <kbd>←</kbd> / <kbd>→</kbd> Navigate within the focused diagram
          </li>
          <li>
            <kbd>Enter</kbd> Inspect the selected component
          </li>
          <li>
            <kbd>Esc</kbd> Close the active card or overlay
          </li>
          <li>
            <kbd>F</kbd> Full page: hide the site header and title
          </li>
        </ul>
      </section>

      <section>
        <h2>Credits</h2>
        <p>
          Created by <a href="https://github.com/nbharathik" target="_blank" rel="noreferrer">
            Bharathi Kannan N
          </a>. The interaction design is
          inspired by{' '}
          <a
            href="https://playground.tensorflow.org/"
            target="_blank"
            rel="noreferrer"
          >
            TensorFlow Playground
          </a>{' '}
          and the{' '}
          <a
            href="https://poloclub.github.io/transformer-explainer/"
            target="_blank"
            rel="noreferrer"
          >
            Poloclub visual explainers
          </a>
          .
        </p>
      </section>
    </div>
  );
}
