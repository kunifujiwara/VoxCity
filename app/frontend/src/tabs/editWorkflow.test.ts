import { describe, expect, it } from 'vitest';
import {
  actionForWorkflow,
  defaultBackdropForTarget,
  defaultWorkflowForTarget,
  drawColorForTarget,
  interactionForWorkflow,
  methodOptionsForTask,
  normalizeWorkflow,
  overlayLabel,
  taskOptionsForTarget,
  TARGET_OPTIONS,
  type EditWorkflowState,
} from './editWorkflow';

function optionIds<T extends { id: string }>(options: T[]) {
  return options.map((option) => option.id);
}

describe('edit workflow options', () => {
  it('lists the three edit targets in the target-first order', () => {
    expect(TARGET_OPTIONS).toEqual([
      { id: 'building', label: 'Building' },
      { id: 'tree', label: 'Tree' },
      { id: 'land_cover', label: 'Land cover' },
    ]);
  });

  it('returns target-specific task options', () => {
    expect(optionIds(taskOptionsForTarget('building'))).toEqual(['add', 'height', 'remove']);
    expect(optionIds(taskOptionsForTarget('tree'))).toEqual(['add', 'remove']);
    expect(optionIds(taskOptionsForTarget('land_cover'))).toEqual(['paint']);
  });

  it('returns task-specific method options', () => {
    expect(optionIds(methodOptionsForTask('building', 'add'))).toEqual(['rectangle', 'polygon']);
    expect(optionIds(methodOptionsForTask('building', 'height'))).toEqual(['click', 'area']);
    expect(optionIds(methodOptionsForTask('building', 'remove'))).toEqual(['click', 'area']);
    expect(optionIds(methodOptionsForTask('tree', 'add'))).toEqual(['click', 'area']);
    expect(optionIds(methodOptionsForTask('tree', 'remove'))).toEqual(['click', 'area']);
    expect(optionIds(methodOptionsForTask('land_cover', 'paint'))).toEqual(['click', 'area']);
  });
});

describe('workflow defaults and normalization', () => {
  it('defaults each target to its first useful task and method', () => {
    expect(defaultWorkflowForTarget('building')).toEqual({ target: 'building', task: 'add', method: 'rectangle' });
    expect(defaultWorkflowForTarget('tree')).toEqual({ target: 'tree', task: 'add', method: 'click' });
    expect(defaultWorkflowForTarget('land_cover')).toEqual({ target: 'land_cover', task: 'paint', method: 'click' });
  });

  it('preserves a valid state', () => {
    const state: EditWorkflowState = { target: 'building', task: 'height', method: 'area' };
    expect(normalizeWorkflow(state)).toEqual(state);
  });

  it('replaces invalid task and method combinations with target defaults', () => {
    expect(normalizeWorkflow({ target: 'tree', task: 'height', method: 'area' })).toEqual({
      target: 'tree',
      task: 'add',
      method: 'click',
    });
    expect(normalizeWorkflow({ target: 'land_cover', task: 'paint', method: 'rectangle' })).toEqual({
      target: 'land_cover',
      task: 'paint',
      method: 'click',
    });
  });
});

describe('workflow action mapping', () => {
  it.each([
    [{ target: 'building', task: 'add', method: 'rectangle' }, 'add_rect', 'draw_rect_3pt'],
    [{ target: 'building', task: 'add', method: 'polygon' }, 'add_polygon', 'draw_polygon'],
    [{ target: 'building', task: 'height', method: 'click' }, 'set_height_click', 'click_feature'],
    [{ target: 'building', task: 'height', method: 'area' }, 'set_height_area', 'draw_polygon'],
    [{ target: 'building', task: 'remove', method: 'click' }, 'remove_click', 'click_feature'],
    [{ target: 'building', task: 'remove', method: 'area' }, 'remove_area', 'draw_polygon'],
    [{ target: 'tree', task: 'add', method: 'click' }, 'add_click', 'click_point'],
    [{ target: 'tree', task: 'add', method: 'area' }, 'add_area', 'draw_polygon'],
    [{ target: 'tree', task: 'remove', method: 'click' }, 'remove_click', 'click_point'],
    [{ target: 'tree', task: 'remove', method: 'area' }, 'remove_area', 'draw_polygon'],
    [{ target: 'land_cover', task: 'paint', method: 'click' }, 'paint_click', 'click_point'],
    [{ target: 'land_cover', task: 'paint', method: 'area' }, 'paint_area', 'draw_polygon'],
  ] as const)('maps %o to %s and %s', (state, expectedAction, expectedInteraction) => {
    expect(actionForWorkflow(state)).toBe(expectedAction);
    expect(interactionForWorkflow(state)).toBe(expectedInteraction);
  });
});

describe('workflow display defaults', () => {
  it('keeps target-specific overlay and draw-color defaults', () => {
    expect(defaultBackdropForTarget('building')).toBe('buildings');
    expect(defaultBackdropForTarget('tree')).toBe('canopy');
    expect(defaultBackdropForTarget('land_cover')).toBe('land_cover');
    expect(drawColorForTarget('building')).toBe('red');
    expect(drawColorForTarget('tree')).toBe('green');
    expect(drawColorForTarget('land_cover')).toBe('blue');
  });

  it('returns concise overlay labels for the 2D panel header', () => {
    expect(overlayLabel('buildings')).toBe('Buildings overlay');
    expect(overlayLabel('canopy')).toBe('Canopy overlay');
    expect(overlayLabel('land_cover')).toBe('Land cover overlay');
    expect(overlayLabel('none')).toBe('No overlay');
  });
});
