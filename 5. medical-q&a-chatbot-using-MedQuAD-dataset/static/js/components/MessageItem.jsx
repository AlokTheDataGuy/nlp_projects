const MessageItem = ({ message }) => {
  const { type, text, entities, entity_labels, abbreviations, source, timestamp } = message;

  // Format timestamp
  const formattedTime = new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  });

  return (
    <div className={`message ${type}`}>
      {type === 'assistant' && (
        <div className="message-avatar">🤖</div>
      )}
      <div className="message-content">
        <p>{text}</p>

        {/* Display entities if available */}
        {entities && entities.length > 0 && (
          <div className="entity-tags">
            <h4>Medical Entities:</h4>
            {entities.map((entity, index) => (
              <span key={index} className="entity-tag">
                {entity} {entity_labels && entity_labels[index] ? `(${entity_labels[index]})` : ''}
              </span>
            ))}
          </div>
        )}

        {/* Display abbreviations if available */}
        {abbreviations && abbreviations.length > 0 && (
          <div className="abbreviation-list">
            <h4>Medical Abbreviations:</h4>
            <ul>
              {abbreviations.map((abbr, index) => (
                <li key={index}>
                  <span className="abbreviation">{abbr.abrv}</span> = <span className="long-form">{abbr.long_form}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Display source if available */}
        {source && (
          <div className="source-citation">
            Source: {source}
          </div>
        )}

        <div className="message-metadata">
          {formattedTime}
        </div>
      </div>
      {type === 'user' && (
        <div className="message-avatar">👤</div>
      )}
    </div>
  );
};
