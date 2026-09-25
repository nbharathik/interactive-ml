/** Loads a preset from the free-play rail and starts the run when the preset asks for it. */

import { useCallback, useEffect, useRef } from 'react';

import type { SimulationApi } from './useSimulation';
import { prefersReducedMotion } from './useSimulation';
import type { ExplainerParamsApi, ParamsRecord } from './useExplainerParams';

/** Long enough for the params change to reset the simulation first. */
const PLAY_DELAY = 60;

export function usePresets<P extends ParamsRecord, S>(
  params: Pick<ExplainerParamsApi<P>, 'applyPreset'>,
  sim: Pick<SimulationApi<S>, 'play' | 'pause'>,
): { pick: (id: string) => void } {
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const pick = useCallback(
    (id: string) => {
      if (timer.current !== null) window.clearTimeout(timer.current);
      sim.pause();
      const preset = params.applyPreset(id);
      if (!preset?.autoRun || prefersReducedMotion()) return;
      timer.current = window.setTimeout(() => {
        timer.current = null;
        sim.play();
      }, PLAY_DELAY);
    },
    [params, sim],
  );

  return { pick };
}
