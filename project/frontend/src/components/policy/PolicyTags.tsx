import React from 'react';

interface PolicyTagsProps {
  tags: string[];
}

const PolicyTags: React.FC<PolicyTagsProps> = ({ tags }) => {
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {tags.map((tag) => (
        <span
          key={tag}
          className="px-3 py-1 bg-sbi-light-purple text-sbi-purple rounded-full text-sm"
        >
          {tag}
        </span>
      ))}
    </div>
  );
};

export default PolicyTags;