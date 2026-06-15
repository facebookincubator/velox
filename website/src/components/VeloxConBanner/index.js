import React from 'react';
import styles from './styles.module.css';
import Link from '@docusaurus/Link';

export default function VeloxConBanner() {
  return (
    <section className={styles.banner} role="region" aria-label="VeloxCon announcement">
      <div className={styles.container}>
        <div className={styles.text}>
          <h2 className={styles.title}>VeloxCon 2026 Recordings Are Now Available</h2>
          <p className={styles.subtitle}>Watch all sessions on-demand on our YouTube channel</p>
        </div>

        <Link
          className={styles.button}
          to="https://www.youtube.com/playlist?list=PLJvBe8nQAEsE6CVJCCx7yMENUS7Id7Ptt"
          target="_blank"
          rel="noopener noreferrer"
        >
          Watch Now →
        </Link>
      </div>
    </section>
  );
}
