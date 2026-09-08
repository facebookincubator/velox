import React from 'react';
import styles from './styles.module.css';
import Link from '@docusaurus/Link';

export default function VeloxConBanner() {
  return (
    <section className={styles.banner} role="region" aria-label="VeloxCon announcement">
      <div className={styles.container}>
        <div className={styles.text}>
          <h2 className={styles.title}>VeloxCon China 2026 — December 5, Shanghai</h2>
          <p className={styles.subtitle}>Alibaba Xuhui Riverside Park, Xuhui District. Join the Velox community in Shanghai.</p>
        </div>

        <Link
          className={styles.button}
          to="https://www.bagevent.com/event/9240454"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn More →
        </Link>
      </div>
    </section>
  );
}
