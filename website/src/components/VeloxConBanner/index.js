import React from 'react';
import styles from './styles.module.css';
import Link from '@docusaurus/Link';

export default function VeloxConBanner() {
  return (
    <section className={styles.banner} role="region" aria-label="VeloxCon China announcement">
      <div className={styles.container}>
        <div className={styles.text}>
          <h2 className={styles.title}>Announcing VeloxCon China 2025</h2>
          <p className={styles.subtitle}>December 13 | Ant Group Offices, Beijing</p>
        </div>
        <Link
          className={styles.button}
          to="https://www.bagevent.com/event/veloxcon-china-2025-en"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn more and Register →
        </Link>
      </div>
    </section>
  );
}
