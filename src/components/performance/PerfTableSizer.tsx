// SPDX-License-Identifier: Apache-2.0
//
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import { Button, ButtonVariant, Icon, Size, Tag } from '@blueprintjs/core';
import { IconNames } from '@blueprintjs/icons';
import classNames from 'classnames';
import { ReactNode, memo } from 'react';
import { PERF_HEURISTIC_FLAG_DEFINITIONS } from '../../definitions/PerfHeuristics';
import { getPerfCellClassName } from '../../functions/perfFunctions';
import { PerfTableColumnSizing, PerfTableSizing, getFlagSetKey } from '../../hooks/usePerfTableColumnSizing';

interface SizerSampleProps {
    children: ReactNode;
}

/** Contributes its content's width to the column while taking no height. */
function SizerSample({ children }: SizerSampleProps) {
    return <div className='perf-table-sizer-sample'>{children}</div>;
}

interface SizerTextProps {
    text: string;
    isNoWrap?: boolean;
}

// The text is generated content, so the sample is invisible to text queries and assistive tech
// while laying out exactly like the cell string it copies.
function SizerText({ text, isNoWrap = false }: SizerTextProps) {
    return (
        <span
            className={classNames('perf-table-sizer-text', { 'is-no-wrap': isNoWrap })}
            data-text={text}
        />
    );
}

interface SizerCellProps {
    sizing: PerfTableColumnSizing;
}

function SizerCell({ sizing }: SizerCellProps) {
    const { column, widestText, longestRunText, isNoWrap, hasIcon, widestMemoryLabel, flagSets, hasNpeLink } = sizing;

    return (
        <td className={getPerfCellClassName(column)}>
            {widestText && (
                <SizerSample>
                    <SizerText
                        text={widestText}
                        isNoWrap={isNoWrap}
                    />
                </SizerSample>
            )}

            {longestRunText && longestRunText !== widestText && (
                <SizerSample>
                    <SizerText
                        text={longestRunText}
                        isNoWrap={isNoWrap}
                    />
                </SizerSample>
            )}

            {hasIcon && (
                <SizerSample>
                    <Icon icon={IconNames.WARNING_SIGN} />
                </SizerSample>
            )}

            {widestMemoryLabel && (
                <SizerSample>
                    <Tag className='memory-tag'>
                        <SizerText text={widestMemoryLabel} />
                    </Tag>
                </SizerSample>
            )}

            {flagSets.map((flags) => (
                <SizerSample key={getFlagSetKey(flags)}>
                    <div className='perf-heuristic-flags'>
                        {flags.map((flag) => (
                            <Tag
                                key={flag}
                                className='perf-heuristic-flag'
                                minimal
                            >
                                <SizerText text={PERF_HEURISTIC_FLAG_DEFINITIONS[flag].shortLabel} />
                            </Tag>
                        ))}
                    </div>
                </SizerSample>
            ))}

            {hasNpeLink && (
                <SizerSample>
                    <Button
                        icon={IconNames.Random}
                        variant={ButtonVariant.MINIMAL}
                        className='graph-button'
                        tabIndex={-1}
                    />
                </SizerSample>
            )}
        </td>
    );
}

interface PerfTableSizerProps {
    sizing: PerfTableSizing;
}

/**
 * A zero-height row holding, per column, the widest content any row in the report renders. A
 * virtualised table only lays out its rendered window, so without it auto table layout would size
 * the columns to whatever is on screen and they would shift as the user scrolls. With it the browser
 * sizes them exactly as it would with every row rendered, wrapping and all.
 */
function PerfTableSizer({ sizing }: PerfTableSizerProps) {
    return (
        <tbody
            className='perf-table-sizer'
            aria-hidden='true'
        >
            <tr>
                <td className='cell'>
                    {sizing.hasTensorTrigger && (
                        <SizerSample>
                            <span className='perf-tensor-trigger-wrapper'>
                                <Button
                                    className='perf-tensor-trigger'
                                    icon={IconNames.FLOW_LINEAR}
                                    variant={ButtonVariant.MINIMAL}
                                    size={Size.SMALL}
                                    tabIndex={-1}
                                />
                            </span>
                        </SizerSample>
                    )}
                </td>
                {sizing.columns.map((columnSizing) => (
                    <SizerCell
                        key={columnSizing.column.key}
                        sizing={columnSizing}
                    />
                ))}
            </tr>

            {sizing.widestAdvice && (
                <tr>
                    <td
                        colSpan={sizing.columns.length + 1}
                        className='cell advice'
                    >
                        <SizerSample>
                            <ul>
                                <li>
                                    <SizerText text={sizing.widestAdvice} />
                                </li>
                                {sizing.longestRunAdvice && sizing.longestRunAdvice !== sizing.widestAdvice && (
                                    <li>
                                        <SizerText text={sizing.longestRunAdvice} />
                                    </li>
                                )}
                            </ul>
                        </SizerSample>
                    </td>
                </tr>
            )}
        </tbody>
    );
}

export default memo(PerfTableSizer);
