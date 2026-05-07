import type { Backdrop, DrawColor, MapInteraction } from '../components/PlanMapEditor';

export type EditTarget = 'building' | 'tree' | 'land_cover';
export type EditTask = 'add' | 'height' | 'remove' | 'paint';
export type EditMethod = 'rectangle' | 'polygon' | 'click' | 'area';

export type ModeAction =
  | 'add_rect'
  | 'add_polygon'
  | 'remove_click'
  | 'remove_area'
  | 'set_height_click'
  | 'set_height_area'
  | 'add_click'
  | 'add_area'
  | 'paint_click'
  | 'paint_area';

export interface EditWorkflowState {
  target: EditTarget;
  task: EditTask;
  method: EditMethod;
}

// ── Configuration tables ──────────────────────────────────

const TASK_OPTIONS: Record<EditTarget, { id: EditTask; label: string; tone?: 'danger' }[]> = {
  building: [
    { id: 'add', label: 'Add' },
    { id: 'height', label: 'Height' },
    { id: 'remove', label: 'Remove', tone: 'danger' },
  ],
  tree: [
    { id: 'add', label: 'Add' },
    { id: 'remove', label: 'Remove', tone: 'danger' },
  ],
  land_cover: [
    { id: 'paint', label: 'Paint' },
  ],
};

const METHOD_OPTIONS: Record<EditTarget, Partial<Record<EditTask, { id: EditMethod; label: string; tone?: 'danger' }[]>>> = {
  building: {
    add: [
      { id: 'rectangle', label: 'Rectangle' },
      { id: 'polygon', label: 'Polygon' },
    ],
    height: [
      { id: 'click', label: 'Click' },
      { id: 'area', label: 'Area' },
    ],
    remove: [
      { id: 'click', label: 'Click', tone: 'danger' },
      { id: 'area', label: 'Area', tone: 'danger' },
    ],
  },
  tree: {
    add: [
      { id: 'click', label: 'Click' },
      { id: 'area', label: 'Area' },
    ],
    remove: [
      { id: 'click', label: 'Click', tone: 'danger' },
      { id: 'area', label: 'Area', tone: 'danger' },
    ],
  },
  land_cover: {
    paint: [
      { id: 'click', label: 'Click' },
      { id: 'area', label: 'Area' },
    ],
  },
};

// ── Exports ───────────────────────────────────────────────

export const TARGET_OPTIONS: { id: EditTarget; label: string }[] = [
  { id: 'building', label: 'Building' },
  { id: 'tree', label: 'Tree' },
  { id: 'land_cover', label: 'Land cover' },
];

export function taskOptionsForTarget(target: EditTarget) {
  return TASK_OPTIONS[target];
}

export function methodOptionsForTask(target: EditTarget, task: EditTask) {
  return METHOD_OPTIONS[target][task] ?? [];
}

export function defaultWorkflowForTarget(target: EditTarget): EditWorkflowState {
  const task = taskOptionsForTarget(target)[0].id;
  const method = methodOptionsForTask(target, task)[0].id;
  return { target, task, method };
}

export function normalizeWorkflow(state: EditWorkflowState): EditWorkflowState {
  const tasks = taskOptionsForTarget(state.target);
  const taskValid = tasks.some((option) => option.id === state.task);
  const task = taskValid ? state.task : tasks[0].id;
  const methods = methodOptionsForTask(state.target, task);
  const method =
    taskValid && methods.some((option) => option.id === state.method)
      ? state.method
      : methods[0].id;
  return { target: state.target, task, method };
}

export function actionForWorkflow(state: EditWorkflowState): ModeAction {
  const normalized = normalizeWorkflow(state);
  const key = `${normalized.target}:${normalized.task}:${normalized.method}`;
  switch (key) {
    case 'building:add:rectangle': return 'add_rect';
    case 'building:add:polygon': return 'add_polygon';
    case 'building:height:click': return 'set_height_click';
    case 'building:height:area': return 'set_height_area';
    case 'building:remove:click': return 'remove_click';
    case 'building:remove:area': return 'remove_area';
    case 'tree:add:click': return 'add_click';
    case 'tree:add:area': return 'add_area';
    case 'tree:remove:click': return 'remove_click';
    case 'tree:remove:area': return 'remove_area';
    case 'land_cover:paint:click': return 'paint_click';
    case 'land_cover:paint:area': return 'paint_area';
    default:
      throw new Error(`Unsupported edit workflow: ${key}`);
  }
}

export function interactionForWorkflow(state: EditWorkflowState): MapInteraction {
  switch (actionForWorkflow(state)) {
    case 'add_rect': return 'draw_rect_3pt';
    case 'add_polygon':
    case 'remove_area':
    case 'set_height_area':
    case 'add_area':
    case 'paint_area':
      return 'draw_polygon';
    case 'remove_click':
    case 'set_height_click':
      return state.target === 'building' ? 'click_feature' : 'click_point';
    case 'add_click':
    case 'paint_click':
      return 'click_point';
  }
}

export function defaultBackdropForTarget(target: EditTarget): Backdrop {
  switch (target) {
    case 'building': return 'buildings';
    case 'tree': return 'canopy';
    case 'land_cover': return 'land_cover';
  }
}

export function drawColorForTarget(target: EditTarget): DrawColor {
  switch (target) {
    case 'building': return 'red';
    case 'tree': return 'green';
    case 'land_cover': return 'blue';
  }
}

export function overlayLabel(backdrop: Backdrop): string {
  switch (backdrop) {
    case 'buildings': return 'Buildings overlay';
    case 'canopy': return 'Canopy overlay';
    case 'land_cover': return 'Land cover overlay';
    case 'none': return 'No overlay';
  }
}
