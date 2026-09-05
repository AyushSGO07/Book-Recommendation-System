// BookVerse Global JS

function addToReadingList(title, author, image) {
    let readingList = JSON.parse(localStorage.getItem('reading_list') || '[]');
    
    // Check if already exists
    const exists = readingList.some(book => book.title === title);
    
    if (!exists) {
        readingList.push({ title, author, image, status: 'want_to_read', dateAdded: new Date().toISOString() });
        localStorage.setItem('reading_list', JSON.stringify(readingList));
        showToast('Added to Reading List');
    } else {
        showToast('Already in your Reading List');
    }
}

function showToast(message) {
    // Remove existing toast if any
    const existing = document.getElementById('bv-toast');
    if (existing) {
        existing.remove();
    }
    
    const toast = document.createElement('div');
    toast.id = 'bv-toast';
    toast.style.position = 'fixed';
    toast.style.bottom = '24px';
    toast.style.right = '24px';
    toast.style.backgroundColor = 'var(--bg-elevated)';
    toast.style.color = 'var(--text-primary)';
    toast.style.padding = '12px 24px';
    toast.style.borderRadius = 'var(--radius-md)';
    toast.style.border = '1px solid var(--border-subtle)';
    toast.style.boxShadow = '0 10px 25px -5px rgba(0, 0, 0, 0.5)';
    toast.style.fontSize = '0.875rem';
    toast.style.fontWeight = '500';
    toast.style.zIndex = '9999';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    toast.innerHTML = `<div style="display: flex; align-items: center; gap: 8px;"><i data-lucide="check-circle" style="width: 16px; height: 16px; color: #10B981;"></i> ${message}</div>`;
    
    document.body.appendChild(toast);
    
    // Initialize icons in toast
    if (window.lucide) {
        lucide.createIcons({ root: toast });
    }
    
    // Animate in
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    }, 10);
    
    // Remove after 3s
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => {
            if (document.body.contains(toast)) {
                toast.remove();
            }
        }, 300);
    }, 3000);
}

// Global Tab handling (for Reading List, etc.)
document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all
                tabs.forEach(t => t.classList.remove('active'));
                // Add to clicked
                tab.classList.add('active');
                
                // (In a real app, we would filter the reading list based on tab text)
            });
        });
    }
});
