import React from 'react';
import styles from './styles.module.css';
import Link from '@docusaurus/Link';

export default function VeloxConBanner() {
  return (
    <section className={styles.banner} role="region" aria-label="VeloxCon announcement">
      <div className={styles.container}>
        <div className={styles.text}>
          <h2 className={styles.title}>Announcing VeloxCon 2026</h2>
          <p className={styles.subtitle}>Meta HQ, Bay Area</p>
          <p className={styles.date}>April 29–30, 2026</p>
        </div>

        <Link
          className={styles.button}
          to="https://veloxcon.io/"
          target="_blank"
          rel="noopener noreferrer"
        >
          Register Now →
        </Link>
      </div>
    </section>
  );
}