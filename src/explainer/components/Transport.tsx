/** The playback controls. Space plays, arrows step, R resets, unless a field or button has focus. */

import { useEffect, useRef, useState } from 'react';

import type { SimulationApi, SpeedName } from '../useSimulation';
import { RUN_AHEAD_STEPS, SPEED_LABELS } from '../useSimulation';
import { readFlag, writeFlag } from '../storage';
import { useLessonTarget } from '../lessonFocus';
import {
  IconChevron,
  IconEnd,
  IconKeyboard,
  IconPause,
  IconPlay,
  IconReplay,
  IconReset,
  IconStep,
} from './Icons';
import { Popover } from './Popover';

const SPEEDS: SpeedName[] = ['slow', 'normal', 'fast', 'turbo'];

/** Set once the visitor has run anything; the play button stops nudging. */
const PLAYED_KEY = 'mlx-played-once';

export interface TransportProps {
  sim: Pick<
    SimulationApi<unknown>,
    | 'isRunning'
    | 'isComplete'
    | 'iteration'
    | 'speed'
    | 'play'
    | 'pause'
    | 'toggle'
    | 'stepOnce'
    | 'reset'
    | 'replay'
    | 'setSpeed'
    | 'runToCompletion'
  >;
  /** Word for one logical step: "epoch", "iteration", "split", "round". */
  unit: string;
  /** Total steps if the run is bounded, renders a progress bar. */
  total?: number;
  /** The status over the buttons once the run has finished. */
  completeLabel?: string;
  /** Its tone. */
  completeTone?: 'good' | 'warn' | 'bad';
  /** What one unit is, as the counter's tooltip. */
  unitHelp?: string;
}

export function Transport({
  sim,
  unit,
  total,
  completeLabel = 'Converged',
  completeTone = 'good',
  unitHelp,
}: TransportProps) {
  const done = sim.isComplete && !sim.isRunning;

  const [played, setPlayed] = useState(() => readFlag(PLAYED_KEY));
  useEffect(() => {
    if (!played && (sim.isRunning || sim.iteration > 0)) {
      setPlayed(true);
      writeFlag(PLAYED_KEY, true);
    }
  }, [played, sim.isRunning, sim.iteration]);
  const nudge = !played && sim.iteration === 0 && !sim.isRunning;

  // One listener for the page's life; it reads the simulation through a ref.
  const simRef = useRef(sim);
  simRef.current = sim;
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      // A widget that used the key (a turntable, a unit picture, a tab list) keeps it.
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable) return;
        // Anything clickable owns Space.
        if (event.code === 'Space' && target.closest('button, summary, a, [role="button"]')) return;
      }
      const s = simRef.current;
      if (event.code === 'Space') {
        event.preventDefault();
        if (s.isComplete && !s.isRunning) s.replay();
        else s.toggle();
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        s.stepOnce(event.shiftKey ? 10 : 1);
      } else if (event.key === 'r' || event.key === 'R') {
        s.reset();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const bounded = typeof total === 'number' && total > 0;
  const progress = bounded ? Math.min(1, sim.iteration / (total as number)) : null;
  const runLabel = bounded
    ? 'Run to the end without animating'
    : 'Run ' + RUN_AHEAD_STEPS.toLocaleString('en-US') + ' ' + unit + 's without animating';

  // Lesson steps target the whole button group; a ring on the play button alone would be clipped.
  const buttons = useLessonTarget('transport', 'buttons');
  const speedTarget = useLessonTarget('transport', 'speed');

  // The label over the buttons: where the run stands, in the same line as the knob labels.
  const status = sim.isComplete ? completeLabel : sim.isRunning ? 'Running' : sim.iteration > 0 ? 'Paused' : 'Ready';

  return (
    <div className="mlx-transport" role="group" aria-label="Playback controls">
      <span className="mlx-toprow__caption">Playback</span>
      <div className="mlx-transport__cells">
        <div className="mlx-transport__cell">
          <span
            className="mlx-transport__label"
            role="status"
            data-tone={sim.isComplete ? completeTone : undefined}
            data-done={sim.isComplete || undefined}
          >
            {status}
          </span>
          <div className="mlx-transport__buttons" {...buttons.attrs}>
            <button
              type="button"
              className="mlx-tbtn"
              onClick={sim.reset}
              title="Reset to the start (R)"
              aria-label="Reset"
            >
              <IconReset size={15} />
            </button>

            <button
              type="button"
              className="mlx-tbtn mlx-tbtn--play"
              data-replay={done || undefined}
              data-nudge={nudge || undefined}
              onClick={done ? sim.replay : sim.toggle}
              aria-label={sim.isRunning ? 'Pause' : done ? 'Replay from the start' : 'Play'}
              title={sim.isRunning ? 'Pause (space)' : done ? 'Replay (space)' : 'Play (space)'}
            >
              {sim.isRunning ? (
                <IconPause size={16} />
              ) : done ? (
                <IconReplay size={16} />
              ) : (
                <IconPlay size={16} />
              )}
            </button>

            <button
              type="button"
              className="mlx-tbtn"
              onClick={() => sim.stepOnce(1)}
              disabled={sim.isComplete}
              title={'Advance one ' + unit + ' (right arrow)'}
              aria-label={'Advance one ' + unit}
            >
              <IconStep size={15} />
            </button>

            <button
              type="button"
              className="mlx-tbtn"
              onClick={() => sim.runToCompletion()}
              disabled={sim.isComplete}
              title={runLabel}
              aria-label={runLabel}
            >
              <IconEnd size={15} />
            </button>
          </div>
        </div>

        <div className="mlx-transport__cell mlx-counter" title={unitHelp}>
          <span className="mlx-transport__label">{unit.charAt(0).toUpperCase() + unit.slice(1)}</span>
          <span className="mlx-counter__value mlx-num">
            {bounded ? sim.iteration + ' / ' + total : sim.iteration.toLocaleString('en-US')}
          </span>
          {progress !== null ? (
            <span className="mlx-counter__progress" aria-hidden="true">
              <span style={{ transform: 'scaleX(' + progress + ')' }} />
            </span>
          ) : null}
        </div>

        <label className="mlx-transport__cell mlx-speed" {...speedTarget.attrs}>
          <span className="mlx-transport__label">Speed</span>
          <div className="mlx-select">
            <select
              value={sim.speed}
              onChange={(event) => sim.setSpeed(event.target.value as SpeedName)}
              aria-label="Playback speed"
              title="Steps per second: slow 0.35x, normal 1x, fast 4x, turbo 20x"
            >
              {SPEEDS.map((speed) => (
                <option key={speed} value={speed}>
                  {SPEED_LABELS[speed]}
                </option>
              ))}
            </select>
            <IconChevron size={12} />
          </div>
        </label>

        <div className="mlx-transport__cell mlx-transport__cell--keys">
          <span className="mlx-transport__label">Keys</span>
          <KeysMenu unit={unit} />
        </div>
      </div>
    </div>
  );
}

/** The shortcut list, behind a small keyboard button. */
function KeysMenu({ unit }: { unit: string }) {
  return (
    <Popover className="mlx-keys-menu" button={<IconKeyboard size={15} />} label="Keyboard shortcuts">
      <ul className="mlx-keys mlx-keys--stack">
        <li>
          <kbd>Space</kbd> play, pause or replay
        </li>
        <li>
          <kbd>→</kbd> one {unit}
        </li>
        <li>
          <kbd>Shift</kbd> + <kbd>→</kbd> ten {unit}s
        </li>
        <li>
          <kbd>R</kbd> reset
        </li>
        <li>
          <kbd>←</kbd> <kbd>→</kbd> move through the diagram when it has focus
        </li>
        <li>
          <kbd>Enter</kbd> open the focused component
        </li>
        <li>
          <kbd>Esc</kbd> close a card or a full-screen view
        </li>
        <li>
          <kbd>F</kbd> full page: hide the site header and title
        </li>
      </ul>
    </Popover>
  );
}
