import React from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import type { RecipeData } from '../../types';
import styles from './RecipePanel.module.css';

export const RecipePanel: React.FC = () => {
    const { activeCard } = useWebSocket();

    // Add tool-focus class to body when recipe is active, removed when not
    // This affects layout in global CSS potentially, but we should handle it here or in App structure.
    // legacy used body class. We can use a prop or context in App layout.
    // For now, let's just render the card.

    if (!activeCard) {
        return (
            <section className={`${styles.recipePanel} ${styles.toolPanel}`}>
                <div className={styles.panelHeader}>
                    <h2 className={styles.panelTitle}>Active Tool</h2>
                    <div className={styles.panelActions}>
                        <span className={styles.panelMeta}>Waiting for tool</span>
                    </div>
                </div>
                <div className={styles.recipeCard}>
                    <div className={styles.noRecipe}>
                        <div className={styles.noRecipeIcon}>
                            <svg viewBox="0 0 24 24">
                                <path d="M3 7c5-3 13-3 18 0l-9 14L3 7z" fill="none" strokeWidth="2" strokeLinejoin="round"></path>
                                <circle cx="10" cy="12" r="1" fill="currentColor"></circle>
                                <circle cx="14" cy="10" r="1" fill="currentColor"></circle>
                            </svg>
                        </div>
                        <div>No tool active</div>
                        <div className={styles.noRecipeHint}>Ask for a recipe or command</div>
                    </div>
                </div>
            </section>
        );
    }

    const { title, subtitle, data, card_type } = activeCard;
    const recipeData = data as RecipeData;
    const style = recipeData.style ? recipeData.style.toUpperCase() : 'RECIPE';

    // Format Tags
    const tags = [];
    if (recipeData.ball_count && recipeData.ball_weight_g) {
        tags.push(`${recipeData.ball_count} x ${Math.round(recipeData.ball_weight_g)}g`);
    }
    if (recipeData.total_dough_g) {
        tags.push(`${Math.round(recipeData.total_dough_g)}g total`);
    }
    if (recipeData.hydration_percent !== undefined) {
        tags.push(`${recipeData.hydration_percent}% hydration`);
    }
    if (recipeData.cold_ferment_hours) {
        tags.push(`${recipeData.cold_ferment_hours}h cold`);
    }
    if (recipeData.bake_temp_f) {
        tags.push(`${recipeData.bake_temp_f}F bake`);
    }

    const formatGrams = (value?: number) => {
        if (value === null || value === undefined || Number.isNaN(value)) return '';
        if (value < 1) return `${value.toFixed(2)}g`;
        if (value < 10) return `${value.toFixed(1)}g`;
        return `${Math.round(value)}g`;
    }

    const updatedAt = recipeData.updated_at ? new Date(recipeData.updated_at) : null;
    const updatedText = updatedAt
        ? `Updated ${updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
        : '';

    return (
        <section className={`${styles.recipePanel} ${styles.toolPanel}`}>
            <div className={styles.panelHeader}>
                <h2 className={styles.panelTitle}>Active Tool</h2>
                <div className={styles.panelActions}>
                    <span className={styles.panelMeta}>{card_type.replace(/_/g, ' ')}</span>
                </div>
            </div>
            <div className={styles.recipeCard}>
                <div className={styles.recipeHeader}>
                    <div>
                        <div className={styles.recipeTitle}>{title}</div>
                        <div className={styles.recipeSubtitle}>{subtitle}</div>
                    </div>
                    <div className={styles.recipePill}>{style}</div>
                </div>

                <div className={styles.recipeTags}>
                    {tags.map((tag, i) => <span key={i} className={styles.recipeTag}>{tag}</span>)}
                </div>

                <div>
                    <div className={styles.recipeSectionTitle}>Ingredients</div>
                    <div className={styles.recipeIngredients}>
                        {recipeData.ingredients && recipeData.ingredients.length > 0 ? (
                            recipeData.ingredients.map((item, i) => (
                                <React.Fragment key={i}>
                                    <div className={styles.ingredientName}>{item.name || 'Ingredient'}</div>
                                    <div className={styles.ingredientAmount}>{formatGrams(item.grams)}</div>
                                    <div className={styles.ingredientPercent}>{item.bakers_percent !== undefined ? `${item.bakers_percent}%` : ''}</div>
                                </React.Fragment>
                            ))
                        ) : (
                            <React.Fragment>
                                <div className={styles.ingredientName}>No ingredients yet</div>
                                <div className={styles.ingredientAmount}></div>
                                <div className={styles.ingredientPercent}></div>
                            </React.Fragment>
                        )}
                    </div>
                </div>

                <div>
                    <div className={styles.recipeSectionTitle}>Steps</div>
                    <div className={styles.recipeSteps}>
                        {recipeData.steps && recipeData.steps.length > 0 ? (
                            recipeData.steps.slice(0, 4).map((step, i) => (
                                <div key={i} className={styles.recipeStep}>
                                    <div className={styles.recipeStepNumber}>{step.number || i + 1}</div>
                                    <div>{step.instruction}</div>
                                </div>
                            ))
                        ) : (
                            <div className={styles.recipeStep}>No steps yet</div>
                        )}
                    </div>
                </div>

                <div className={styles.recipeUpdated}>{updatedText}</div>
            </div>
        </section>
    );
};
