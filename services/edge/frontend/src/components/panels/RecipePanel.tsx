import React, { useEffect } from 'react';
import { useWebSocket } from '../../contexts/WebSocketContext';
import type { CalendarCardEvent, RecipeData } from '../../types';
import styles from './RecipePanel.module.css';

export const RecipePanel: React.FC = () => {
  const { activeCard } = useWebSocket();

  useEffect(() => {
    document.body.classList.toggle('tool-focus', Boolean(activeCard));
    return () => {
      document.body.classList.remove('tool-focus');
    };
  }, [activeCard]);

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
              <svg viewBox="0 0 24 24" aria-hidden="true">
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

  const typeLabel = activeCard.card_type.replace(/_/g, ' ');

  return (
    <section className={`${styles.recipePanel} ${styles.toolPanel}`}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Active Tool</h2>
        <div className={styles.panelActions}>
          <span className={styles.panelMeta}>{typeLabel}</span>
        </div>
      </div>
      {activeCard.card_type === 'recipe' ? renderRecipeCard(activeCard) : null}
      {activeCard.card_type === 'calendar' ? renderCalendarCard(activeCard) : null}
      {activeCard.card_type !== 'recipe' && activeCard.card_type !== 'calendar' ? renderGenericCard(activeCard) : null}
    </section>
  );
};

const renderRecipeCard = (card: any) => {
  const { title, subtitle, data } = card;
  const recipeData = data as RecipeData;
  const style = recipeData.style ? recipeData.style.toUpperCase() : 'RECIPE';

  const tags: string[] = [];
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
  };

  const updatedAt = recipeData.updated_at ? new Date(recipeData.updated_at) : null;
  const updatedText = updatedAt
    ? `Updated ${updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
    : '';

  return (
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
            recipeData.steps.slice(0, 5).map((step, i) => (
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
  );
};

const renderCalendarCard = (card: any) => {
  const events = Array.isArray(card?.data?.events) ? (card.data.events as CalendarCardEvent[]) : [];

  return (
    <div className={styles.recipeCard}>
      <div className={styles.recipeHeader}>
        <div>
          <div className={styles.recipeTitle}>{card.title || 'Calendar'}</div>
          <div className={styles.recipeSubtitle}>{card.subtitle || 'Upcoming events'}</div>
        </div>
        <div className={styles.recipePill}>AGENDA</div>
      </div>

      <div className={styles.calendarList}>
        {events.length === 0 ? (
          <div className={styles.calendarEmpty}>No events in this range.</div>
        ) : (
          events.map((event, index) => (
            <article key={`${event.title}_${event.start}_${index}`} className={styles.calendarEvent}>
              <div className={styles.calendarTime}>{formatCalendarTime(event.start, event.end)}</div>
              <div className={styles.calendarTitle}>{event.title || 'Untitled event'}</div>
              {event.location ? <div className={styles.calendarLocation}>{event.location}</div> : null}
            </article>
          ))
        )}
      </div>
    </div>
  );
};

const renderGenericCard = (card: any) => {
  const dataPreview = (() => {
    try {
      return JSON.stringify(card.data ?? {}, null, 2);
    } catch {
      return '{}';
    }
  })();

  return (
    <div className={styles.recipeCard}>
      <div className={styles.recipeHeader}>
        <div>
          <div className={styles.recipeTitle}>{card.title || 'Tool Result'}</div>
          <div className={styles.recipeSubtitle}>{card.subtitle || card.card_type}</div>
        </div>
        <div className={styles.recipePill}>{(card.card_type || 'tool').toUpperCase()}</div>
      </div>

      {card.body ? <p className={styles.genericBody}>{card.body}</p> : null}
      <pre className={styles.genericData}>{dataPreview}</pre>
    </div>
  );
};

const formatCalendarTime = (startIso?: string, endIso?: string) => {
  if (!startIso) {
    return 'Unknown time';
  }

  try {
    const start = new Date(startIso);
    if (!endIso) {
      return start.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
    }
    const end = new Date(endIso);
    return `${start.toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    })} - ${end.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
  } catch {
    return startIso;
  }
};
