/** The studio frame shown while a deep link's chunk loads: the markup index.html paints before React runs. */

const html =
  typeof document === 'undefined' ? '' : (document.getElementById('mlx-skeleton')?.innerHTML ?? '');

export function PageSkeleton() {
  return <div role="status" aria-label="Loading" dangerouslySetInnerHTML={{ __html: html }} />;
}
