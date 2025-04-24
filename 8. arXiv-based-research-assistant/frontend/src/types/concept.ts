export interface Concept {
  concept_id: number;
  name: string;
  definition: string;
  paper_id: string;
}

export interface ConceptRelation {
  source_concept_id: number;
  target_concept_id: number;
  relation_type: string;
  source_name: string;
  target_name: string;
}
